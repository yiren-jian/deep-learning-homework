from fully_connected import FullyConnected
import torch


def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (72 x 2), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    # X = dataset["X"]
    # W = dataset["W"]
    # B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE

    X = torch.randn(48,2, requires_grad=True, dtype=torch.float64)
    W = torch.randn(72,2, requires_grad=True, dtype=torch.float64)
    B = torch.randn(72,1, requires_grad=True, dtype=torch.float64)
    y = full_connected(X, W, B)
    l = torch.randn(y.size(), dtype=torch.float64)
    z = torch.nn.functional.mse_loss(y, l)
    dzdy, = torch.autograd.grad(outputs=z,inputs=y)
    y.backward(dzdy)
    dzdx_a = X.grad
    dzdw_a = W.grad
    dzdb_a = B.grad

    err = {}
    is_correct = torch.autograd.gradcheck(full_connected, inputs=(X,W,B), eps=DELTA, atol=TOL)
    print(is_correct)
    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)
