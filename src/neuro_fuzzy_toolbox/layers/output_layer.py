import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    """
    Output layer for an Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Aggregates the weighted rule outputs by summing across all rules and applies an output activation function to produce 
    the final model prediction. The activation is determined by the specified output type:

    - ``'default'``: No activation (identity). Suitable for regression.
    - ``'sigmoid'``: Sigmoid activation..
    - ``'softmax'``: Softmax activation (applied only when ``return_probs=True`` in :meth:`forward`). Suitable for classification.
    """
    def __init__(self, output_type):
        """
        Initializes a new OutputLayer instance.

        Args:
            output_type (str): Output activation type. Must be one of ``'default'``, ``'sigmoid'``, or ``'softmax'``.
        """
        super(OutputLayer, self).__init__()
        self._output_type = output_type.lower()
        
        if (self._output_type == 'default' or self._output_type == 'softmax'):
            self._last_layer = nn.Identity()
        elif (self._output_type == 'sigmoid'):
            self._last_layer = nn.Sigmoid()
            
        if self._output_type != 'softmax':
            self._get_output = lambda rules_outputs, return_probs: torch.sum(rules_outputs, dim=-1).t().squeeze(1)
        else:
            self._get_output = lambda rules_outputs, return_probs: nn.functional.softmax(torch.sum(rules_outputs, dim=-1).t().squeeze(1), dim=1) if return_probs else torch.sum(rules_outputs, dim=-1).t().squeeze(1)
        

    def forward(self, rules_outputs, return_probs=False):
        """
        Forward pass of the output layer.

        Sums the weighted rule outputs across all rules and applies the configured output activation.

        Args:
            rules_outputs (torch.Tensor): Weighted rule outputs of shape ``(outputs, batch_size, rules)``.
            return_probs (bool): If ``True`` and the output type is ``'softmax'``, applies a softmax activation 
                to return class probabilities. Ignored for all other output types. Defaults to ``False``.

        Returns:
            torch.Tensor: Model output of shape ``(batch_size, outputs)``, or ``(batch_size,)`` for single-output models.
        """
        return self._last_layer(self._get_output(rules_outputs, return_probs))