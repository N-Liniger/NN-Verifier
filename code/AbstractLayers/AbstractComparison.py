import torch
from AbstractBounds import BackBounds, Bounds

class AbstractComparison(torch.nn.Module):
    '''
        AbstractLinear:
            This is always the last layer and compares the intervals of the final layers. Thereby, one does backsub all the way
            back to the beginning again. In the constructor the comparison weight matrix is created
        forward:
            Passes the final result.
    '''
    def __init__(self, prev: torch.nn.Module = None, true_label: int = None) -> None:
        super().__init__()
        self.prev = prev
        self.true_label = true_label
        self.inshape = self.prev.weights.shape[0]
        self.weights = torch.zeros(self.inshape - 1, self.inshape)
        self.weights[:, self.true_label] = 1
        diag_matrix = -1 * torch.eye(self.inshape)
        diag_matrix = torch.cat((diag_matrix[:self.true_label], diag_matrix[self.true_label + 1:]))
        self.weights += diag_matrix
    

    def forward(self, bounds: Bounds):
        backBounds = BackBounds(lowerBoundWeights = self.weights.detach().clone(),
                                upperBoundWeights = self.weights.detach().clone(),
                                upperBoundBias = torch.zeros(self.inshape - 1),
                                lowerBoundBias = torch.zeros(self.inshape - 1))
        return self.prev.backsubstitution(backBounds)