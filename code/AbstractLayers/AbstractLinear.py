import torch
from AbstractBounds import BackBounds, Bounds

class AbstractLinear(torch.nn.Module):
    '''
        AbstractLinear: Abstraction of the linear layer
        backsubstitution:
            Functions the same as all the previous backsubstitution but one thing to notice is that the linear transformer is tight
            which means that when calculating the upperBoundBias,lowerBoundBias,lowerBoundWeights,upperBoundWeights we always use the same
            self.weights and self.bias
        forward:
            Passes the bounds resulting from the backsubstitution
    '''
    def __init__(self, layer: torch.nn.modules.linear.Linear, prev: torch.nn.Module) -> None:
        super().__init__()
        self.prev = prev
        self.weights = layer.weight.detach()
        self.bias = layer.bias.detach()
    
    def backsubstitution(self, backBounds: BackBounds) -> Bounds:
        #Note: in comparison to ReLU linear layers are tights and therefore there is no need to check for positive/negative weights
        backBounds.upperBoundBias += backBounds.upperBoundWeights @ self.bias
        backBounds.lowerBoundBias += backBounds.lowerBoundWeights @ self.bias
        backBounds.upperBoundWeights = backBounds.upperBoundWeights @ self.weights
        backBounds.lowerBoundWeights = backBounds.lowerBoundWeights @ self.weights
        return self.prev.backsubstitution(backBounds)
        
    def forward(self, bounds: Bounds) -> Bounds:
        size = self.weights.size(0)
        backBounds = BackBounds(upperBoundWeights = self.weights.detach().clone(),
                                lowerBoundWeights = self.weights.detach().clone(),
                                upperBoundBias = self.bias.detach().clone(),
                                lowerBoundBias = self.bias.detach().clone())
        return self.prev.backsubstitution(backBounds)