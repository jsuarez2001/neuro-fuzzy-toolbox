import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, output_type):
        super(OutputLayer, self).__init__()
        self._output_type = output_type.lower()
        
        if (self._output_type == 'regression' or self._output_type == 'multiclass'):
            self._last_layer = nn.Identity()
        elif (self._output_type == 'binary'):
            self._last_layer = nn.Sigmoid()

    def forward(self, x, return_probabilities=False):
        x = torch.sum(x, dim=-1).t().squeeze(1)
        if return_probabilities and self._output_type == 'multiclass':
            x = nn.functional.softmax(x, dim=1)
        return self._last_layer(x)