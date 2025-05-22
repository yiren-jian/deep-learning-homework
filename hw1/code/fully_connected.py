import torch


class FullyConnected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        """
        Computes the output of the fully_connected function given in the assignment

        Arguments
        ---------
        ctx: a PyTorch context object
        x (Tensor): of size (T x n), the input features
        w (Tensor): of size (m x n), the weights
        b (Tensor): of size (m), the biases

        Returns
        -----
        y (Tensor): of size (T x m), the outputs of the fully_connected operator
        """
        ctx.save_for_backward(x, w, b)
        y = torch.mm(x,w.t()) + b

        return y

    @staticmethod
    def backward(ctx, dz_dy):
        """
        back-propagates the gradients with respect to the inputs
        ctx: a PyTorch context object.
        dz_dy (Tensor): of size (T x m), the gradients with respect to the output argument y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdw (Tenor): of size (m x n), the gradients with respect to w
        dzdb (Tensor): of size (m x 1), the gradients with respect to w
        """
        x, w, b = ctx.saved_tensors

        dzdx = torch.mm(dz_dy, w)
        dzdw = torch.mm(dz_dy.t(), x)
        dzdb = dz_dy.sum(0).squeeze(0)

        return dzdx, dzdw, dzdb
