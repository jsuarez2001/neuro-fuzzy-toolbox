import torch.nn as nn

class FiringLevelsLayer(nn.Module):
    """
    Class for calculating firing levels in an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Methods:
    - __init__: Initializes a new FiringLevelsLayer instance.
    - forward: Performs a forward pass to calculate firing levels.

    Example Usage:
    >>> firing_levels_layer = FiringLevelsLayer()
    >>> firing_levels = firing_levels_layer(membership_values) #assuming 'membership_values' is the tensor obtained from the Fuzzification Layer

    """
    def __init__(self):
        """
        Initializes a new FiringLevelsLayer instance.

        """
        super(FiringLevelsLayer, self).__init__()


    def forward(self, m):
        """
        Performs a forward pass through the layer to calculate firing levels.

        Parameters:
        - m (torch.Tensor): Input tensor containing the membership values for each rule.

        Returns:
        - torch.Tensor: Output tensor (Firing levels).

        """
        w = m.prod(dim=m.dim()-2)
        return w