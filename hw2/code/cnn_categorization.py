from create_dataset import create_dataset
from cnn_categorization_base import cnn_categorization_base
from train import train
from torch import random, save
from argparse import ArgumentParser

random.manual_seed(0)


def cnn_categorization(model_type, data_path, contrast_normalization, whiten):
    output_path = "{}_image_categorization_dataset.pt".format(model_type)
    exp_dir = "./{}_models".format(model_type)
    train_ds, val_ds = create_dataset(data_path, output_path, contrast_normalization, whiten)

    if model_type == "base":
        # create netspec_opts
        netspec_opts = {}
        netspec_opts['kernel_size'] = [(3,3), None, None, (3,3), None, None, (3,3), None, None, (8,8), (1,1)]
        netspec_opts['num_filters'] = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64, 16]
        netspec_opts['stride'] = [1, None, None, 2, None, None, 2, None, None, 1, 1]
        netspec_opts['padding'] = [1, None, None, 1, None, None, 1, None, None, 0, 0]
        netspec_opts['layer_type'] = ['conv', 'bn', 'relu', 'conv', 'bn', 'relu', 'conv', 'bn', 'relu', 'pool', 'pred']
        # create train_opts
        train_opts = {}
        train_opts["batch_size"] = 128
        train_opts["lr"] = 0.1
        train_opts["momentum"] = 0.9
        train_opts["weight_decay"] = 0.0001
        train_opts["step_size"] = 20
        train_opts["gamma"] = 0.1
        train_opts["num_epochs"] = 40

        model = cnn_categorization_base(netspec_opts)

    elif model_type == "advanced":
        # create netspec_opts
        pass
        # create train_opts

        # uncomment the line below if you have implemented cnn_categorization_advanced and remember to import it
        # model =  cnn_categorization_advanced(netspec_opts)
    else:
        raise ValueError(f"Error: unknown model type {model_type}")

    model = train(model, train_ds, val_ds, train_opts, exp_dir)
    save(model, "{}-model.pt".format(model_type))


if __name__ == '__main__':

    # change the defaults to your preferred parameter values
    # You can also  pass in values from the terminal
    # for example, to change model type from base to advanced
    # type <cnn_categorization.py --model_type advanced> at a terminal and press enter
    args = ArgumentParser()
    args.add_argument("--model_type", type=str, default="base", required=False,
                      help="The model type must be either base or advanced")
    args.add_argument("--data_path", type=str, default="image_categorization_dataset.pt",
                      required=False, help="specify the dataset for the training")
    args.add_argument("--contrast_normalization", type=bool, default=False, required=False,
                      help="Specify the contrast_normalization value")
    args.add_argument("--whiten", type=bool, default=False, required=False,
                      help="Specify whether to whiten value")
    args, _ = args.parse_known_args()
    cnn_categorization(**args.__dict__)
