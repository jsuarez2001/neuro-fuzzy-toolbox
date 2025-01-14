import torch

def weighted_linear(x, c, w):
    """
    Compute the weighted linear combination of input features and coefficients (representing the
    consequent parameters of the ANFIS model). It is manufactured in such a way that it also works with
    more than one input (batches) and will be used to calculate the individual outputs of each rule of
    the ANFIS model.

    Parameters:
    - x (torch.Tensor): Input data tensor.
    - c (torch.Tensor): Coefficients tensor for the linear combination (consequent parameters).
    - w (torch.Tensor): Weights tensor for element-wise multiplication.

    Returns:
    - torch.Tensor: Resulting weighted linear combination.

    Explanation:
    This function calculates the weighted linear combination of the input features `x` and the coefficients `c`.
    It involves matrix multiplication of the input features by the transposed coefficients (excluding the last column),
    adding the last column of coefficients, and then element-wise multiplication by the weights `w`.
    The formula used is:

    \\[ (x \\cdot c[:, :-1].T + c[:, -1]) \\cdot w \\]

    Note:
    - The weights `w` are applied element-wise.

    """
    return (torch.bmm(x.unsqueeze(0).expand(c[:, :, :-1].size(0), -1, -1), torch.transpose(c[:, :, :-1], 1, 2)) + c[:, :, -1].unsqueeze(1)).mul(w.unsqueeze(0))