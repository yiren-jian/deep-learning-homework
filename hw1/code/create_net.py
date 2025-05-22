from torch import nn
from generalized_logistic_layer import GeneralizedLogisticLayer
from fully_connected_layer import FullyConnectedLayer


def create_net(input_features, hidden_units, non_linearity, output_size):
    """
    Constructs a model based on the specifications passed as input arguments

    Arguments
    --------
    input_features (integer): The number of input features
    hidden_units (List): of length L where L is the number of hidden layers. hidden_units[i] denotes the
                        number of units at hidden layer i + 1 for i  = 0, ..., L - 1
    non_linearity (List):  of length L. non_linearity[i] contains a string describing the type of non-linearity to use
                           hidden layer i + 1 for i = 0, ... L-1
    output_size (integer): The number of units in the output layer

    Returns
    -------
    sequential_net (Sequential): the constructed model
    """
    num_hidden_layer = len(hidden_units)
    layers = []
    for layer in range(num_hidden_layer):
        if layer==0:
            layers.append(FullyConnectedLayer(input_features, hidden_units[layer]))
        else:
            layers.append(FullyConnectedLayer(hidden_units[layer-1], hidden_units[layer]))
        layers.append(GeneralizedLogisticLayer(non_linearity[layer]))
    layers.append(FullyConnectedLayer(hidden_units[-1], output_size))

    sequential_net = nn.Sequential(*layers)
    return sequential_net
