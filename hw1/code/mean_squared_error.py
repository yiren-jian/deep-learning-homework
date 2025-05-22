import torch


class MeanSquaredError(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        """
        computes the mean squared error between x1 and x2

        Arguments
        -------
        ctx: a pytorch context object
        x1 (Tensor): of size (T x n), inputs
        x2 (Tensor): of size (T x n), targets

        Returns
        ------
        y (scalar): The mean squared error
        """
        ctx.save_for_backward(x1, x2)
        y = torch.mean((x1 - x2) * (x1 - x2))

        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagates the error with respect to the input arguments

        Arguments
        --------
        ctx: A PyTorch context object
        dzdy (Tensor): of size(1), the gradient with respect to y

        Returns
        ------
        dzdx1 (Tensor): of size(T x n), the gradients w.r.t X1
        dzdx2 (Tensor): of size(T x n), the gradients w.r.t X2
        """

        x1, x2 = ctx.saved_tensors
        T = x1.size(0)
        n = x1.size(1)
        dzdx1 = dzdy * (2/(T*n)) * (x1-x2)
        dzdx2 = dzdy * (2/(T*n)) * (x2-x1)

        return dzdx1, dzdx2
