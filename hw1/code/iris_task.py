from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE %%%

# Set the data_path
data_path = 'iris_dataset.pt'

# set pre-processing options
mean_subtraction = False
normalization = False

iris_dataset = load_dataset(data_path, mean_subtraction, normalization)

# %% set the input arguments for the create net
input_features = 4
hidden_units = [16, 12]
non_linearity = ['tanH', 'tanH']
output_size = 3

# create a sequential model and assign it to variable net
net = create_net(input_features, hidden_units, non_linearity, output_size)

# specify train_opts
train_opts = {}
train_opts['lr'] = 0.01
train_opts['momentum'] = 0.9
train_opts['weight_decay'] = 0.0001
train_opts['step_size'] = 40
train_opts['gamma'] = 1
train_opts['batch_size'] = 24
train_opts['num_epochs'] = 80


# Train and save the trained model
net = train(net, iris_dataset, train_opts)
save({'state_dict': net.state_dict()}, "iris_solution.pt")
