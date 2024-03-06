import torch
from AbstractLayers.AbstractLinear import AbstractLinear
from AbstractLayers.AbstractInput import AbstractInput
from AbstractLayers.AbstractComparison import AbstractComparison
from AbstractLayers.AbstractRelu import AbstractRelu
from AbstractLayers.AbstractLeakyReluLess import AbstractLeakyReluLess
from AbstractLayers.AbstractLeakyReluMore import AbstractLeakyReluMore
from AbstractLayers.AbstractConvLayer import AbstractConvLayer
from AbstractBounds import Bounds


class AbstractNetwork(torch.nn.Module):
    '''
    AbstractNetwork: 
        takes in a neural network and constructs an equivalent abstract network
        that propagates the bounds
    constructor:
        Takes in the NN and constructs a nn.Sequential containing the abstract layers
    forward:
        just calls forward on the sequential
    '''
    def __init__(self, net: torch.nn.Sequential, true_label: int, sample=None) -> None:
        super().__init__()


        self.true_label = true_label
        abstractLayers = [AbstractInput()]

        consumed_layers = []

        for i, layer in enumerate(net):



            if type(layer) == torch.nn.modules.linear.Linear:
                abstractLayers.append(AbstractLinear(layer, prev=abstractLayers[-1]))

            elif type(layer) == torch.nn.modules.flatten.Flatten:
                pass

            elif type(layer) == torch.nn.modules.ReLU:
                abstractLayers.append(AbstractRelu(layer, prev=abstractLayers[-1]))

            elif type(layer) == torch.nn.modules.LeakyReLU:
                #Check if slope of LeakyRelu is less or more than 1
                if layer.negative_slope <= 1:
                    abstractLayers.append(AbstractLeakyReluLess(layer, prev=abstractLayers[-1]))
                elif layer.negative_slope > 1:
                    abstractLayers.append(AbstractLeakyReluMore(layer, prev=abstractLayers[-1]))

            elif type(layer) == torch.nn.Conv2d:
                if sample is None:
                    raise Exception("Error in AbstractNetwork constructor: sample is None, but Conv2d layer needs sample input")
                out = torch.nn.Sequential(*consumed_layers)(sample)
                # out is the input to a conv2d layer, so it must have the form (channels, height, width)
                abstractLayers.append(AbstractConvLayer(layer, prev=abstractLayers[-1], inChannels=out.shape[0], inHeight=out.shape[1], inWidth=out.shape[2]))

            else:
                raise Exception(f"Error in AbstractNetwork constructor: {type(layer)} is not supported")

            consumed_layers.append(layer)

        abstractLayers.append(AbstractComparison(prev=abstractLayers[-1], true_label=true_label))
        self.sequential = torch.nn.Sequential(*abstractLayers)

    def forward(self, bounds: Bounds) -> Bounds:
        return self.sequential(bounds)

