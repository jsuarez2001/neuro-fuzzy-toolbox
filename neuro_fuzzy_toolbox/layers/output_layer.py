import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, output_type):
        super(OutputLayer, self).__init__()
        _output_type = output_type.lower()
        
        if (_output_type == 'regression' or _output_type == 'multiclass'):
            self._last_layer = nn.Identity()
        elif (_output_type == 'binary'):
            self._last_layer = nn.Sigmoid()

    def forward(self, x):
        x = torch.sum(x, dim=-1).t().squeeze(1)
        return self._last_layer(x)