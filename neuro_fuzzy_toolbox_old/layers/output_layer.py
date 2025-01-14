import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    """
    Class for representing the last layer (output layer) of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Methods:
    - __init__: Initializes a new OutputLayer instance.
    - forward: Performs a forward pass to calculate the final output.

    Example Usage:
    >>> output_layer = OutputLayer()
    >>> output = output_layer(rule_outputs) # Assuming rule_outputs is the tensor obtained from the consequent layer with shape (batch_size, rules)

    """
    def __init__(self):
        """
        Initializes a new OutputLayer instance.

        """
        super(OutputLayer, self).__init__()

    def forward(self, x):
        """
        Performs a forward pass to calculate the final output by computing the sum along the last dimension of
        the input tensor.

        Parameters:
        - x (torch.Tensor): Input tensor (Rule outputs).

        Returns:
        - torch.Tensor: Final output.

        """
        return torch.sum(x, dim=-1)