from torch import nn


def cnn_categorization_base(netspec_opts):
    """
    Constructs a sequential model

    Arguments
    --------
    netspec_opts (Dictionary): Contains the network's specifications. It has the keys
                 'kernel_size', 'num_filters', 'stride', 'padding', and 'layer_type'.
                 Each of these keys point to a list containing the corresponding property of each layer.
    Return
    -----
     sequential_net (nn.Sequential):  The constructed model
    """

    # instantiate nn.Sequential
    sequential_net = nn.Sequential()
    # add the hidden layers to the network as specified in netspec_opts
    num_layers = len(netspec_opts['layer_type'])
    for i in range(num_layers):
        layer_name = '_'.join([netspec_opts['layer_type'][i], str(i)])
        in_channels = 3 if i==0 else netspec_opts['num_filters'][i-1]
        num_filters = netspec_opts['num_filters'][i]
        kernel_size = netspec_opts['kernel_size'][i]
        stride = netspec_opts['stride'][i]
        padding = netspec_opts['padding'][i]
        if netspec_opts['layer_type'][i] == 'conv' and i != num_layers-1:
            sequential_net.add_module(layer_name, nn.Conv2d(in_channels, num_filters, kernel_size, stride, padding))
        elif netspec_opts['layer_type'][i] == 'bn':
            sequential_net.add_module(layer_name, nn.BatchNorm2d(num_filters))
        elif netspec_opts['layer_type'][i] == 'relu':
            sequential_net.add_module(layer_name, nn.ReLU(inplace=True))
        elif netspec_opts['layer_type'][i] == 'pool':
            sequential_net.add_module(layer_name, nn.AvgPool2d(kernel_size))

        if i == num_layers-1:
            sequential_net.add_module('pred', nn.Conv2d(in_channels, num_filters, kernel_size, stride, padding))

    return sequential_net
