import torch
from AbstractBounds import BackBounds, Bounds

class AbstractRelu(torch.nn.Module):
    '''
        AbstractRelu: Abstraction of the ReLU layer
        backsubstitution:
            The transformer is non tight so need to check for the sign of the weights
        forward:
            Passes the bounds computed by means of the backsubstitution
    '''
    def __init__(self, layer: torch.nn.modules.ReLU, prev: torch.nn.Module) -> None:
        super().__init__()
        self.prev = prev
        self.one = torch.tensor([1.])
        self.zero = torch.tensor([0.])
        self.alpha = torch.nn.Parameter(torch.ones(prev.weights.shape[0])/2, requires_grad=True)

    def backsubstitution(self, backBounds: BackBounds) -> Bounds:
        # Precompute the conditions
        upper_weights_positive = torch.max(backBounds.upperBoundWeights, self.zero)
        upper_weights_negative = torch.min(backBounds.upperBoundWeights, self.zero)
        lower_weights_positive = torch.max(backBounds.lowerBoundWeights, self.zero)
        lower_weights_negative = torch.min(backBounds.lowerBoundWeights, self.zero)

        # Update the biases
        backBounds.upperBoundBias += upper_weights_positive @ self.upperBoundBias + upper_weights_negative @ self.lowerBoundBias
        backBounds.lowerBoundBias += lower_weights_positive @ self.lowerBoundBias + lower_weights_negative @ self.upperBoundBias

        # Update the weights
        backBounds.upperBoundWeights = upper_weights_positive @ self.upperBoundWeights + upper_weights_negative @ self.lowerBoundWeights
        backBounds.lowerBoundWeights = lower_weights_positive @ self.lowerBoundWeights + lower_weights_negative @ self.upperBoundWeights

        return self.prev.backsubstitution(backBounds)


    def forward(self, bounds: Bounds) -> Bounds:

        #Negative: a priori assume that all the intervalls are fully negative
        self.lowerBoundWeights = torch.zeros_like(bounds.lowerBound)
        self.upperBoundWeights = torch.zeros_like(bounds.upperBound)
        self.upperBoundBias = torch.zeros_like(bounds.upperBound)
        self.lowerBoundBias = torch.zeros_like(bounds.lowerBound)

        #Positive: should be one on the diagonal if the interval is fully positive
        positive = bounds.lowerBound > self.zero
        self.lowerBoundWeights = torch.where(positive, self.one, self.lowerBoundWeights)
        self.upperBoundWeights = torch.where(positive, self.one, self.upperBoundWeights)

        #Crossing: set upper bound mult to slope, lower bound mult to alpha, adjust upper bound bias accordingly
        crossing = torch.logical_and((bounds.lowerBound < self.zero) , (bounds.upperBound > self.zero))
        slope = torch.where(crossing, torch.div(bounds.upperBound, bounds.upperBound - bounds.lowerBound), self.zero)
        self.upperBoundWeights = torch.where(crossing, slope, self.upperBoundWeights)
        self.upperBoundBias = torch.where(crossing, - slope * bounds.lowerBound, self.upperBoundBias)

        #self.lowerBoundWeights = torch.where(crossing, torch.sigmoid(self.alpha), self.lowerBoundWeights)
        self.alpha.data.clamp_(min=0, max=1)
        self.lowerBoundWeights = torch.where(crossing, self.alpha, self.lowerBoundWeights)


        #Convert the upperBoundWeights and  lowerBoundWeights into a diagonal tensor
        self.lowerBoundWeights = torch.diag(self.lowerBoundWeights.squeeze())
        self.upperBoundWeights = torch.diag(self.upperBoundWeights.squeeze())
        self.upperBoundBias.squeeze_()
        self.lowerBoundBias.squeeze_()

        backBounds = BackBounds(lowerBoundWeights = self.lowerBoundWeights.detach().clone(),
                                upperBoundWeights = self.upperBoundWeights.detach().clone(),
                                upperBoundBias = self.upperBoundBias.detach().clone(),
                                lowerBoundBias = self.lowerBoundBias.detach().clone())
        return self.prev.backsubstitution(backBounds)