import argparse
import torch
from AbstractNetwork import AbstractNetwork
from AbstractBounds import Bounds
from networks import get_network
from utils.loading import parse_spec

from tqdm import tqdm
import time
# import matplotlib.pyplot as plt
#from torchviz import make_dot

DEVICE = "cpu"


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:

    start_time = time.time()

    abstractNetwork = AbstractNetwork(net=net, true_label=true_label, sample=inputs)
    InitialUpperBound = (inputs + eps).clamp_(min=0, max=1)
    InitialLowerBound = (inputs - eps).clamp_(min=0, max=1)
    inputBounds = Bounds(InitialLowerBound, InitialUpperBound)
    outputBounds = abstractNetwork.forward(inputBounds)

    #print(outputBounds.lowerBound, outputBounds.upperBound)
    #print(list(abstractNetwork.sequential.parameters()))

    if len(list(abstractNetwork.sequential.parameters())) > 0:
        verified = False
        optimizer = torch.optim.Adam(abstractNetwork.sequential.parameters(), lr=1)
        
        losses = []

        while (time.time() - start_time < 60):
            if outputBounds.lowerBound.min() > 0:
                print(f"---- TIME TAKEN {time.time() - start_time}s ----")
                print(outputBounds.lowerBound.min())
                print(len(losses))
                # plt.plot(losses)
                # plt.show()
                return True
            loss = torch.sum(torch.relu(-outputBounds.lowerBound))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputBounds = abstractNetwork.forward(inputBounds)
            losses.append(outputBounds.lowerBound.min().item())

        # plt.plot(losses)
        # plt.show()

        
        # for i in range(20):
        #     if outputBounds.lowerBound.min() > 0:
        #         return True
        #     loss = torch.sum(-outputBounds.lowerBound)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     outputBounds = abstractNetwork.forward(inputBounds)
            
            # iteration += 1
            #graph = make_dot(loss)
            #graph.render('computation_graph.png', format='png')  # This saves the graph as a PNG file
            # if iteration%50==0:
            #      print(outputBounds.lowerBound, outputBounds.upperBound)

    else:
        if outputBounds.lowerBound.min() > 0:
            return True
        else:
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
