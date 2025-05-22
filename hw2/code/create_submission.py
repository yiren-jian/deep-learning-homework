from torch import load, save
from torch.nn import Softmax
from os.path import exists
from zipfile import ZipFile
from sys import argv


def create_submission(model_type):
    """
    Evaluates the model on the test and validation data and creates a submission file

    Arguments
    ---------
    model_type (string): Specifies the model type for which the submission is being made for.
    """
    data_path = "{}_image_categorization_dataset.pt".format(model_type)
    model_path = "{}-model.pt".format(model_type)

    if not exists(data_path):
        raise ValueError("Error: the data file {} does not exits".format(data_path))
    if not exists(model_path):
        raise ValueError("Error: the trained model {} does not exits".format(model_path))

    dataset = load(data_path)
    model = load(model_path)

    data_te = dataset["data_te"]
    sets_tr = dataset["sets_tr"]
    data_val = dataset["data_tr"]
    data_val = data_val[sets_tr == 2]

    model.eval()
    soft_max = Softmax(dim=1)
    # test set
    prob_test = soft_max(model(data_te).squeeze())
    if prob_test.size() != (9600, 16):
        raise ValueError(f"Expected test set of size (9600, 16) but got size {prob_test.size()} instead")

    # validation
    prob_val = soft_max(model(data_val).squeeze())
    if prob_val.size() != (6400, 16):
        raise ValueError(f"Expected validation set of size (6400, 16) but got size {prob_val.size()} instead")

    output_name_zip = f"./{model_type}_submission.zip"
    output_name_test = f"./{model_type}_testing.pt"
    output_name_val = f"./{model_type}_validation.pt"
    save(prob_test, output_name_test)
    save(prob_val, output_name_val)
    with ZipFile(output_name_zip, 'w') as zipf:
        zipf.write(model_path)
        zipf.write(output_name_test)
        zipf.write(output_name_val)


if __name__ == '__main__':
    m_type = "base"
    try:
        m_type = argv[1]
    except IndexError:
        print("Setting model type to <base> since no type is given.")
        print("To create a submission for advanced model, execute <create_submission.py advanced> at a terminal")
    if m_type not in ("base", "advanced"):
        raise ValueError(f"Model type must be either base or advanced. Got {m_type} instead")
    create_submission(m_type)
