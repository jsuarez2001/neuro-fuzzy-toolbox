import torch

def gaussian2(x, p):
    '''
    Calculates Gaussian values based on the input data and premise parameters of the ANFIS model.
    It is used as its membership function. It is manufactured in such a way that it also works with
    more than one input (batches).
    
    :param x: Tensor of input data.
    :type x: torch.Tensor
    :param p: 3D parameter tensor containing 'mu' and 'sigma' parameters by data dimension and by ANFIS rule.
    :type p: torch.Tensor
    
    :return: Tensor with resulting Gaussian values.
    :rtype: torch.Tensor

    **Explanation:**
    This function calculates Gaussian values based on the input data `x` and the parameters `p`.
    It ensures that the standard deviation values in `p` are not zero by replacing them with a small
    value (1e-6). The formula used to compute the Gaussian values is:

    .. note::
        If `p[:, :, 1]` is zero, it is replaced with 1e-6 to prevent division by zero.

    '''
    return torch.exp(-0.5 * torch.pow((x - p[:, :, 0])/torch.where(p[:, :, 1] == 0, torch.tensor(1e-6), p[:, :, 1]), 2))



def gaussian3(x, p):
    '''
    Calculates Gaussian values based on the input data and premise parameters of the ANFIS model.
    It is used as its membership function. It is manufactured in such a way that it also works with
    more than one input (batches).
    
    :param x: Tensor of input data.
    :type x: torch.Tensor
    :param p: 3D parameter tensor containing 'mu' and 'sigma' parameters by data dimension and by ANFIS rule.
    :type p: torch.Tensor
    
    :return: Tensor with resulting Gaussian values.
    :rtype: torch.Tensor

    **Explanation:**
    This function calculates Gaussian values based on the input data `x` and the parameters `p`.
    It ensures that the standard deviation values in `p` are not zero by replacing them with a small
    value (1e-6). The formula used to compute the Gaussian values is:

    .. note::
        If `p[:, :, 1]` is zero, it is replaced with 1e-6 to prevent division by zero.

    '''
    return p[:, :, 2] * torch.exp(-torch.pow((x - p[:, :, 0])/torch.where(p[:, :, 1] == 0, torch.tensor(1e-6), p[:, :, 1]), 2))



def weighted_linear(x, c, w):
    """
    Compute the weighted linear combination of input features and coefficients (representing the
    consequent parameters of the ANFIS model). It is manufactured in such a way that it also works with
    more than one input (batches) and will be used to calculate the individual outputs of each rule of
    the ANFIS model.

    :param x: Tensor of input data.
    :type x: torch.Tensor
    :param c: Coefficients tensor for the linear combination (consequent parameters).
    :type c: torch.Tensor
    :param w: Weights tensor for element-wise multiplication.
    :type w: torch.Tensor

    :return: Resulting weighted linear combination.
    :rtype: torch.Tensor

    **Explanation:**
    This function calculates the weighted linear combination of the input features `x` and the coefficients `c`.
    It involves matrix multiplication of the input features by the transposed coefficients (excluding the last column),
    adding the last column of coefficients, and then element-wise multiplication by the weights `w`.

    .. note::
        The weights `w` are applied element-wise.

    """
    return (x @ c[:, :-1].t() + c[:, -1]).mul(w)




