import torch

def gaussian2(x, p):
    '''
    Calculates Gaussian values ​​based on the input data and premise parameters of the ANFIS model.
    It is used as its membership function. It is manufactured in such a way that it also works with
    more than one input (batches).

    Parameters:
    - x (torch.Tensor): Tensor of 2D or 3D input data (for single or batch input respectively).
    - p (torch.Tensor): 3D parameter tensor containing 'mu' and 'sigma' parameters by data dimension
                        and by ANFIS rule.

    Returns:
    - torch.Tensor: 2D or 3D tensor with Resulting Gaussian values (for single or batch input
                    respectively).

    Explanation:
    This function calculates Gaussian values based on the input data `x` and the parameters `p`.
    It ensures that the standard deviation values in `p` are not zero by replacing them with a small
    value (1e-6). The formula used to compute the Gaussian values is:

    \\[ \\exp\\left(-0.5 \\cdot \\left(\\frac{x - p[:, :, 0]}{p[:, :, 1]}\\right)^2\\right) \\]

    Note:
    - If `p[:, :, 1]` is zero, it is replaced with 1e-6 to prevent division by zero.

    '''
    return torch.exp(-0.5 * torch.pow((x - p[:, :, 0])/torch.where(p[:, :, 1] == 0, torch.tensor(1e-6), p[:, :, 1]), 2))