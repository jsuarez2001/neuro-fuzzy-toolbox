import torch.nn as nn

from abc import abstractmethod

class ConsequentFunction(nn.Module):
    @abstractmethod
    def forward(self, x, consequents, weights):
        pass
    
    @abstractmethod
    def initialize_consequents(self, outputs, consequents_rules, input_size, input_dtype):
        pass