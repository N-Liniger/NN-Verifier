import torch
from AbstractBounds import BackBounds, Bounds

class AbstractLeakyReluMore(torch.nn.Module):
    '''
        See AbstractRelu.
        The only difference here is how lowerBoundWeights, upperBoundWeights, lowerBoundBias and upperBoundBias
        are computed. Alpha optimization is on the upper bound.
    '''
    def __init__(self, layer: torch.nn.modules.ReLU, prev: torch.nn.Module) -> None:
        super().__init__()
        self.prev = prev
        self.negative_slope = layer.negative_slope
        self.one = torch.tensor([1.])
        self.zero = torch.tensor([0.])
        self.alpha = torch.nn.Parameter((torch.ones(prev.weights.shape[0])*self.negative_slope + 1)/2, requires_grad=True)

    def backsubstitution(self, backBounds: BackBounds):
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

        #Negative: a priori assume that all the intervalls are fully negative set all the elements on the diagonal to negative slope
        self.lowerBoundWeights = torch.ones_like(bounds.lowerBound) * self.negative_slope
        self.upperBoundWeights = torch.ones_like(bounds.upperBound) * self.negative_slope
        self.upperBoundBias = torch.zeros_like(bounds.upperBound)
        self.lowerBoundBias = torch.zeros_like(bounds.lowerBound)

        #Positive: should be one on the diagonal if the interval is fully positive
        positive = bounds.lowerBound > self.zero
        self.lowerBoundWeights = torch.where(positive, self.one, self.lowerBoundWeights)
        self.upperBoundWeights = torch.where(positive, self.one, self.upperBoundWeights)

        #Crossing: upperBoundWeights: set it to alpha, upperBoundBias: 0, lowerBoundWeights: to the slope of lower line, lowerBoundBias: set to intercept
        crossing = torch.logical_and((bounds.lowerBound < self.zero) , (bounds.upperBound > self.zero))
        slope = torch.where(crossing, torch.div(bounds.upperBound - self.negative_slope * bounds.lowerBound, bounds.upperBound - bounds.lowerBound), self.zero)
        self.lowerBoundWeights = torch.where(crossing, slope, self.lowerBoundWeights)
        self.lowerBoundBias = torch.where(crossing, bounds.lowerBound * (self.negative_slope - slope), self.lowerBoundBias)
        
        
        #self.upperBoundWeights = torch.where(crossing, torch.sigmoid(self.alpha) * (self.negative_slope - self.one) + self.one, self.upperBoundWeights)
        self.alpha.data.clamp_(min=1, max=self.negative_slope)
        self.upperBoundWeights = torch.where(crossing, self.alpha, self.upperBoundWeights)

        #Convert the upperBoundWeights and lowerBoundWeights into a diagonal tensor
        self.lowerBoundWeights = torch.diag(self.lowerBoundWeights.squeeze())
        self.upperBoundWeights = torch.diag(self.upperBoundWeights.squeeze())
        self.upperBoundBias.squeeze_()
        self.lowerBoundBias.squeeze_()

        backBounds = BackBounds(lowerBoundWeights = self.lowerBoundWeights.detach().clone(),
                                upperBoundWeights = self.upperBoundWeights.detach().clone(),
                                upperBoundBias = self.upperBoundBias.detach().clone(),
                                lowerBoundBias = self.lowerBoundBias.detach().clone())

        return self.prev.backsubstitution(backBounds)