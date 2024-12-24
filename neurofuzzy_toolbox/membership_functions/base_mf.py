import torch.nn as nn

from abc import abstractmethod

class MembershipFunction(nn.Module):
    def __init__(self):
        super(MembershipFunction, self).__init__()
        self._params = None
        self._simple_implementation = None # Simple numpy implementation needed for plotting
        
    @abstractmethod
    def forward(self, x, premises):
        pass
    
    @abstractmethod
    def initialize_premises(self):
        pass