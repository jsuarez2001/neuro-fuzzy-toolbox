import torch
import torch.nn as nn

class NormalizationLayer(nn.Module):
    """
    Class for normalize the firing levels in an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Methods:
    - __init__: Initializes a new NormalizeLayer instance.
    - forward: Performs a forward pass to normalize the firing levels obtained on a previous layer.

    Example Usage:
    >>> normalization_layer = NormalizationLayer()
    >>> norm_firing_levels = normalization_layer(firing_levels) #assuming 'firing_levels' is the tensor obtained from the Firing Levels Layer

    """
    def __init__(self):
        """
        Initializes a new FiringLevelsLayer instance.

        """
        super(NormalizationLayer, self).__init__()


    def forward(self, w):
        """
        Performs a forward pass through the layer to normalize the firing levels.

        Parameters:
        - x (torch.Tensor): Input tensor containing the membership values for each rule.

        Returns:
        - torch.Tensor: Output tensor (Normalized Firing levels).

        """
        sum = torch.sum(w, dim=1, keepdim=True)
        sum[sum == 0] = 1
        w = w/sum
        return w