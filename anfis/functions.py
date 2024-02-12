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

    \[ \exp\left(-0.5 \cdot \left(\frac{x - p[:, :, 0]}{p[:, :, 1]}\right)^2\right) \]

    Note:
    - If `p[:, :, 1]` is zero, it is replaced with 1e-6 to prevent division by zero.

    '''
    return torch.exp(-0.5 * torch.pow((x - p[:, :, 0])/torch.where(p[:, :, 1] == 0, torch.tensor(1e-6), p[:, :, 1]), 2))



def gaussian3(x, p):
    '''
    Calculates Gaussian values ​​based on the input data and premise parameters of the ANFIS model.
    It is used as its membership function. It is manufactured in such a way that it also works with
    more than one input (batches).
    
    Parameters:
    - x (torch.Tensor): Tensor of 2D or 3D input data (for single or batch input respectively).
    - p (torch.Tensor): 3D parameter tensor containing 'mu', 'sigma' and 'f' parameters by data 
                        dimension and by ANFIS rule.
    
    Returns:
    - torch.Tensor: 2D or 3D tensor with Resulting Gaussian values (for single or batch input 
                    respectively).

    Explanation:
    This function calculates Gaussian values based on the input data `x` and the parameters `p`.
    It ensures that the standard deviation values in `p` are not zero by replacing them with a small
    value (1e-6). The formula used to compute the Gaussian values is:

    \[ \exp\left(-p[:, :, 2] \cdot \left(\frac{x - p[:, :, 0]}{p[:, :, 1]}\right)^2\right) \]

    Note:
    - If `p[:, :, 1]` is zero, it is replaced with 1e-6 to prevent division by zero.

    '''
    return p[:, :, 2] * torch.exp(-torch.pow((x - p[:, :, 0])/torch.where(p[:, :, 1] == 0, torch.tensor(1e-6), p[:, :, 1]), 2))



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

    \[ (x \cdot c[:, :-1].T + c[:, -1]) \cdot w \]

    Note:
    - The weights `w` are applied element-wise.

    """
    return (x @ c[:, :-1].t() + c[:, -1]).mul(w)



def sum(x):
    """
    Calculate the sum along the last dimension of the input tensor. It will be used to compute the final output
    of the ANFIS model.

    Parameters:
    - x (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Sum along the last dimension.

    Explanation:
    This function computes the sum of elements along the last dimension of the input tensor `x`.
    The `dim=-1` argument specifies the last dimension.
    
    """
    return torch.sum(x, dim=-1)



