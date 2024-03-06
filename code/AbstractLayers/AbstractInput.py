import torch
from AbstractBounds import BackBounds, Bounds

class AbstractInput(torch.nn.Module):
    '''
        AbstractInput:
            This is the first layer of every network!
        backsubstitution:
            In this method, the result of the backsubstitution is computed. This means that the input bounds (eps box) are multiplied
            with the lower and upper bound weights which were propagated all the way back.
        forward:
            Flattens the bounds and stores them in an attribute as they will be used for the bound computation of every layer after.
            Then passses the flattened bounds on
    '''

    def __init__(self) -> None:
        super().__init__()

    def backsubstitution(self, backBounds: BackBounds) -> Bounds:

        upperBound = torch.where(backBounds.upperBoundWeights > 0, backBounds.upperBoundWeights, 0) @ self.upperBound +\
            torch.where(backBounds.upperBoundWeights < 0, backBounds.upperBoundWeights, 0) @ self.lowerBound + backBounds.upperBoundBias

        lowerBound = torch.where(backBounds.lowerBoundWeights > 0, backBounds.lowerBoundWeights, 0) @ self.lowerBound +\
            torch.where(backBounds.lowerBoundWeights < 0, backBounds.lowerBoundWeights, 0) @ self.upperBound + backBounds.lowerBoundBias

        # assert torch.sum(upperBound < lowerBound) == 0, "The lower bound exceeds the upperbound"

        return Bounds(lowerBound, upperBound)

    def forward(self, bounds: Bounds) -> Bounds:
        #Assign it here because we need it in the backsub above
        self.lowerBound = torch.flatten(bounds.lowerBound)
        self.upperBound = torch.flatten(bounds.upperBound)
        return Bounds(self.lowerBound, self.upperBound)
    