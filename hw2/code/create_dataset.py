import torch
from torch.utils.data import TensorDataset


def create_dataset(data_path, output_path=None, contrast_normalization=False, whiten=False):
    """
    reads and organizes the dataset into Tensor dataset. Also apply optional
    pre-processing steps to the data

    Arguments
    --------
    data_path (String): The path to the dataset to read
    output_path (String): The name of the file to save the preprocessed data to
    contrast_normalization (boolean): Specifies whether to normalize the data or not. Default (False)
    whiten (boolean): specifies whether to whiten the data or not. Default (False)

    Return
    ------
    train_ds (TensorDataset): A tensor dataset of the training examples (inputs and labels) bundled together
    val_ds (TensorDataset): A tensor dataset of the validation examples (inputs and labels) bundled together
    """
    dataset = torch.load(data_path)
    data_tr = dataset["data_tr"]
    sets_tr = dataset["sets_tr"]
    label_tr = dataset["label_tr"]
    data_te = dataset["data_te"]

    # write the code for the necessary pre-processes below the if block
    if data_path == "image_categorization_dataset.pt":
        # do mean centering on data_tr and data_te
        data_mean = torch.mean(data_tr, dim=0)
        data_tr -= data_mean
        data_te -= data_mean

        # %%% DO NOT EDIT BELOW
        if contrast_normalization:
            image_std = torch.std(data_tr[sets_tr == 1], unbiased=True)
            image_std[image_std == 0] = 1
            data_tr = data_tr / image_std
            data_te = data_te / image_std
        if whiten:
            examples, rows, cols, channels = data_tr.size()
            data_tr = data_tr.view(examples, -1)
            W = torch.matmul(data_tr[sets_tr == 1].T, data_tr[sets_tr == 1]) / examples
            E, V = torch.eig(W, eigenvectors=True)
            en = torch.sqrt(torch.mean(E[:, 0]).squeeze())
            M = torch.diag(en / torch.max(torch.sqrt(E[:, 0].squeeze()), torch.tensor([10.0])))

            data_tr = torch.matmul(data_tr.mm(V.T), M.mm(V))
            data_tr = data_tr.view(examples, rows, cols, channels)

            data_te = data_te.view(-1, rows * cols * channels)
            data_te = torch.matmul(data_te.mm(V.T), M.mm(V))
            data_te = data_te.view(-1, rows, cols, channels)

        preprocessed_data = {"data_tr": data_tr, "data_te": data_te, "sets_tr": sets_tr, "label_tr": label_tr}
        if output_path:
            torch.save(preprocessed_data, output_path)

    train_ds = TensorDataset(data_tr[sets_tr == 1], label_tr[sets_tr == 1])
    val_ds = TensorDataset(data_tr[sets_tr == 2], label_tr[sets_tr == 2])

    return train_ds, val_ds

# %%%% DO NOT EDIT ABOVE
