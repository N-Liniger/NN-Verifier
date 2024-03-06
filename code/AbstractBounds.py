import torch

class BackBounds:
    '''
        BackBounds is passed back from every layer up to the input layer sequentially creating the weights and biases
        used in the computation of the lower and upper bounds of the layers
    '''
    def __init__(self, lowerBoundWeights: torch.Tensor, upperBoundWeights: torch.Tensor,\
                    lowerBoundBias: torch.Tensor, upperBoundBias: torch.Tensor) -> None:
        self.lowerBoundWeights = lowerBoundWeights
        self.upperBoundWeights = upperBoundWeights                               
        self.lowerBoundBias = lowerBoundBias
        self.upperBoundBias = upperBoundBias

class Bounds:
    '''
        This is the object that is passed forward by the network, it simply stores a lower and upper bound
    '''
    def __init__(self, lowerBound: torch.Tensor, upperBound: torch.Tensor) -> None:
        self.lowerBound = lowerBound
        self.upperBound = upperBound