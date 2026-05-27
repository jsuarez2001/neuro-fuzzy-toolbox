import torch
import torch.nn as nn

from abc import abstractmethod

class ConsequentFunction(nn.Module):
    """
    Abstract base class for consequent functions.

    Defines the interface required for implementing consequent functions within the toolbox. 
    Intended as a guide for future implementations.
    """
    @abstractmethod
    def forward(self, x, consequents, weights):
        """Computes the weighted rule outputs for a given batch of inputs."""
        pass
    
    @abstractmethod
    def get_consequents_outputs(self, x, consequents):
        """Returns the unweighted output of each rule for a given batch of inputs."""
        pass
    
    @abstractmethod
    def random_consequents(self, outputs, rules, input_size, dtype):
        """Initializes and returns randomly generated consequent parameters."""
        pass


class Linear_CF(ConsequentFunction):
    """
    Linear consequent function.

    Computes the output of the neuro-fuzzy network from the input features and rule consequents as a linear combination, defined as:

    .. math::

        O_j = \\sum_{i=1}^{n} (c_{i,j} \\cdot x_i) + c_{n+1,j}

    where:
        - :math:`O_j` is the :math:`j`-th output of the consequent layer of an ANFIS model (associated to the :math:`j`-th rule).
        - :math:`x_i` is the :math:`i`-th feature of an input sample :math:`x` of size :math:`n`.
        - :math:`c_{i,j}` is the :math:`i`-th consequent parameter associated with the :math:`j`-th rule of an ANFIS model.

    """
    def forward(self, x, consequents, weights):
        """            
        Forward pass of the linear consequent function.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_size)`` containing the input features.
            consequents (torch.Tensor): Tensor of shape ``(outputs, rules, input_size + 1)`` containing the consequent
                parameters, where ``rules`` is the number of fuzzy rules and ``outputs`` is the number of model outputs.
            weights (torch.Tensor): Tensor of shape ``(batch_size, rules)`` containing the normalized firing levels for each rule.

        Returns:
            torch.Tensor: Tensor of shape ``(outputs, batch_size, rules)`` containing the weighted rule outputs.
        
        """
        return (torch.bmm(x.unsqueeze(0).expand(consequents[:, :, :-1].size(0), -1, -1), torch.transpose(consequents[:, :, :-1], 1, 2)) + consequents[:, :, -1].unsqueeze(1)).mul(weights.unsqueeze(0))
    
    def get_consequents_outputs(self, x, consequents):
        """        
        Returns the individual rule outputs without weighting by normalized firing levels.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_size)`` containing the input features.
            consequents (torch.Tensor): Tensor of shape ``(outputs, rules, input_size + 1)`` containing the consequent
                parameters, where ``rules`` is the number of fuzzy rules and ``outputs`` is the number of model outputs.

        Returns:
            torch.Tensor: Tensor of shape ``(outputs, batch_size, rules)`` containing the unweighted output of each rule, without 
            multiplication by the normalized firing levels.
        """
        with torch.no_grad():
            outputs = torch.bmm(x.unsqueeze(0).expand(consequents[:, :, :-1].size(0), -1, -1), torch.transpose(consequents[:, :, :-1], 1, 2)) + consequents[:, :, -1].unsqueeze(1)
        return outputs

    def random_consequents(self, outputs, rules, input_size, dtype):
        """
        Initializes the consequent parameters randomly in the range ``[-1, 1]``.

        Args:
            outputs (int): Number of model outputs.
            rules (int): Number of fuzzy rules.
            input_size (int): Number of input features.
            dtype (torch.dtype): Data type for the returned tensor.

        Returns:
            torch.Tensor: Tensor of shape ``(outputs, rules, input_size + 1)`` containing randomly initialized consequent parameters.
        """
        return 2 * torch.rand(outputs, rules, input_size + 1, dtype=dtype) - 1