import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, output_type):
        super(OutputLayer, self).__init__()
        _output_type = output_type.lower()
        
        if (_output_type == 'regression'):
            self._last_layer = nn.Identity()
        elif (_output_type == 'binary'):
            self._last_layer = nn.Sigmoid()
        elif (_output_type == 'multiclass'):
            self._last_layer = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.sum(x, dim=-1).t()
        return self._last_layer(x)