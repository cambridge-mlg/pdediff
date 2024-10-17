import ot
import torch
from torch import Tensor


def emd(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
) -> Tensor:
    r"""Computes the earth mover's distance (EMD) between two distributions.

    Wikipedia:
        https://wikipedia.org/wiki/Earth_mover%27s_distance

    Arguments:
        x: A set of samples :math:`x ~ p(x)`.
        y: A set of samples :math:`y ~ q(y)`.
    """

    return ot.emd2(
        x.new_tensor(()),
        y.new_tensor(()),
        torch.cdist(x.flatten(1), y.flatten(1)),
    )


def mmd(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
) -> Tensor:
    r"""Computes the empirical maximum mean discrepancy (MMD) between two distributions.

    Wikipedia:
        https://wikipedia.org/wiki/Kernel_embedding_of_distributions

    Arguments:
        x: A set of samples :math:`x ~ p(x)`.
        y: A set of samples :math:`y ~ q(y)`.
    """

    x = x.flatten(1)
    y = y.flatten(1)

    xx = x @ x.T
    yy = y @ y.T
    xy = x @ y.T

    dxx = xx.diag().unsqueeze(1)
    dyy = yy.diag().unsqueeze(0)

    err_xx = dxx + dxx.T - 2 * xx
    err_yy = dyy + dyy.T - 2 * yy
    err_xy = dxx + dyy - 2 * xy

    mmd = 0

    for sigma in (1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3):
        kxx = torch.exp(-err_xx / sigma)
        kyy = torch.exp(-err_yy / sigma)
        kxy = torch.exp(-err_xy / sigma)

        mmd = mmd + kxx.mean() + kyy.mean() - 2 * kxy.mean()

    return mmd
