import math
import torch
from AbstractBounds import BackBounds, Bounds


class AbstractConvLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Conv2d, prev: torch.nn.Module, inChannels, inHeight, inWidth) -> None:
        super().__init__()

        self.prev = prev

        self.weights = layer.weight.detach()
        self.bias = layer.bias.detach()

        self.padding = layer.padding
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.fs = layer.kernel_size
        self.outChannels = layer.out_channels
        self.inChannels = layer.in_channels
        self.inHeight = inHeight
        self.inWidth = inWidth
        self.outHeight = math.floor((self.inHeight + 2 * self.padding[0] - self.dilation[0] * (self.fs[0] - 1) - 1) / self.stride[0] + 1)
        self.outWidth = math.floor((self.inWidth + 2 * self.padding[1] - self.dilation[1] * (self.fs[1] - 1) - 1) / self.stride[1] + 1)

        # assert self.inChannels == inChannels
        # Only works if padding mode is zero
        # assert layer.padding_mode == 'zeros'

        # Create the matrix, column by column
        weights_matrix = []
        bias_vector = torch.zeros((self.outChannels, self.outHeight, self.outWidth))
        for outChannelIdx in range(self.outChannels):
            for outHeightIdx in range(self.outHeight):
                for outWidthIdx in range(self.outWidth):
                    bias_vector[outChannelIdx, outHeightIdx, outWidthIdx] = self.bias[outChannelIdx]
                    # Create the column
                    row = torch.zeros((self.inChannels, self.inHeight+2*self.padding[0], self.inWidth+2*self.padding[1]))
                    inHeightIdx = outHeightIdx * self.stride[0]
                    inWidthIdx = outWidthIdx * self.stride[1]
                    row[:, inHeightIdx:inHeightIdx + self.fs[0], inWidthIdx:inWidthIdx + self.fs[1]] = self.weights[outChannelIdx]
                    # Remove padding
                    row = row[:, self.padding[0]:self.padding[0] + self.inHeight, self.padding[1]:self.padding[1] + self.inWidth]
                    # Flatten the column
                    row = torch.reshape(row, (self.inChannels * self.inHeight * self.inWidth,))
                    
                    weights_matrix.append(row)

        self.weights = torch.stack(weights_matrix, dim=0)
        self.bias = torch.flatten(bias_vector)


    def backsubstitution(self, backBounds: BackBounds) -> Bounds:
        backBounds.upperBoundBias = backBounds.upperBoundBias + backBounds.upperBoundWeights @ self.bias
        backBounds.lowerBoundBias = backBounds.lowerBoundBias + backBounds.lowerBoundWeights @ self.bias
        backBounds.upperBoundWeights = backBounds.upperBoundWeights @ self.weights
        backBounds.lowerBoundWeights = backBounds.lowerBoundWeights @ self.weights
        return self.prev.backsubstitution(backBounds)

    def forward(self, bounds: Bounds) -> Bounds:
        backBounds = BackBounds(upperBoundWeights=self.weights.detach().clone(),
                                lowerBoundWeights=self.weights.detach().clone(),
                                upperBoundBias=self.bias.detach().clone(),
                                lowerBoundBias=self.bias.detach().clone())
        return self.prev.backsubstitution(backBounds)