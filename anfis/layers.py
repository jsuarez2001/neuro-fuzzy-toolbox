import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import functions as f

class FuzzifyLayer(nn.Module):
    '''
    Fuzzification layer of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Attributes:
    - input_size (int): The size of the input features.
    - init_rules (int): The number of initial fuzzy rules.
    - mf (function): Membership function to use (default: gaussian2).
    - params (list): List of parameter names for the membership function (default: ['mu', 'sigma']).
    - premises (torch.nn.Parameter): Trainable parameters for the fuzzification layer.

    Methods:
    - __init__: Initializes a new FuzzifyLayer instance.
    - init_premises: Initializes the premises based on input training data.
    - forward: Performs a forward pass through the fuzzification layer.
    - premises_structure: Prints the structure of the fuzzy premises.

    Example Usage:
    >>> fuzzify_layer = FuzzifyLayer(input_size=4, init_rules=3)
    >>> input_data = torch.randn((3, 4))
    >>> membership_values = fuzzify_layer(input_data)

    '''
    def __init__(self, input_size, dtype=torch.double, init_rules=1, mf=f.gaussian2, params=['mu', 'sigma'], x_train=[]):
        """
        Initializes a new FuzzifyLayer instance.

        Parameters:
        - input_size (int): The size of the input features.
        - dtype (torch.dtype): Data type for the premises (default: torch.double).
        - init_rules (int): The number of initial fuzzy rules (default: 1).
        - mf (function): Membership function to use (default: gaussian2).
        - params (list): List of parameter names for the membership function (default: ['mu', 'sigma']).
        - x_train (torch.Tensor): Training data for initializing fuzzy premises (default: empty list).

        """
        super(FuzzifyLayer, self).__init__()
        self.input_size = input_size
        self.init_rules = init_rules
        self.mf = mf
        self.params = params

        # Initialize premises
        if x_train == []:
            prems = 2 * torch.rand(input_size, init_rules, len(params), dtype=dtype) - 1
            prems[:,:,1] = (prems[:,:,1] + 1)/2
            self.premises = Parameter(prems, requires_grad=True)
        else:
            self.premises = Parameter(self.init_premises(x_train), requires_grad=True)


    def init_premises(self, x_train):
        """
        Initializes the fuzzy premises based on input training data.

        Parameters:
        - x_train (torch.Tensor): Training data for initializing fuzzy premises.

        Returns:
        - torch.Tensor: Initialized fuzzy premises.

        """
        premises = torch.zeros(self.input_size, self.init_rules, len(self.params), dtype=x_train.dtype)
        if self.init_rules > 1:
            min = torch.min(x_train, dim=0).values
            max = torch.max(x_train, dim=0).values
            stp = (max - min) / (self.init_rules - 1)
            for i in range(self.input_size):
                h = torch.arange(min[i], max[i] + stp[i], stp[i])
                premises[i, :, 0] = h[:self.init_rules]
                premises[i, :, 1] = stp[i]/2
        else:
            for i in range(self.input_size):
                premises[i, :, 0] = torch.mean(x_train[:, i])
                premises[i, :, 1] = torch.std(x_train[:, i])
        return premises


    def forward(self, x):
        """
        Performs a forward pass through the fuzzification layer.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor (membership values).

        """
        return self.mf(x.unsqueeze(x.dim()), self.premises)


    @property
    def premises_structure(self):
        """
        Prints the structure of the fuzzy premises.

        """
        print("Premises Structure:")
        for i in range(self.init_rules):
            print(f"    rule {i + 1}:")
            for j in range(self.input_size):
                print(f"        x{j} parameters:")
                [print(f"            {self.params[k]}: {self.premises[j, i, k]}") for k in range(len(self.params))]



class FiringLevelsLayer(nn.Module):
    """
    Class for calculating firing levels in an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Methods:
    - __init__: Initializes a new FiringLevelsLayer instance.
    - forward: Performs a forward pass to calculate firing levels.

    Example Usage:
    >>> 

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
        - torch.Tensor: Firing levels.

        """
        w = m.prod(dim=m.dim()-2)
        return w


class NormalizationLayer(nn.Module):
    """
    Class for normalize the firing levels in an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Methods:
    - __init__: Initializes a new NormalizeLayer instance.
    - forward: Performs a forward pass to normalize the firing levels obtained on a previous layer.

    Example Usage:
    >>> 

    """
    def __init__(self):
        """
        Initializes a new FiringLevelsLayer instance.

        """
        super(NormalizationLayer, self).__init__()


    def forward(self, w, rules):
        """
        Performs a forward pass through the layer to normalize the firing levels.

        Parameters:
        - x (torch.Tensor): Input tensor containing the membership values for each rule.
        - rules (int): Number of rules in the system.

        Returns:
        - torch.Tensor: Firing levels after normalization.

        """
        if rules == 1:
            w = torch.where(w != 0, 1, w)
        else:
            sum = torch.sum(w, dim=1, keepdim=True)
            sum[sum == 0] = 1
            w = w/sum
        return w


class ConsequentLayer(nn.Module):
    """
    Class for representing the fourth layer (consequent layer) of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Methods:
    - __init__: Initializes a new ConsequentLayer instance.
    - forward: Performs a forward pass to calculate the consequent layer output.
    - consequents_structure: Prints the structure of the consequent parameters.

    Example Usage:
    >>> consequent_layer = ConsequentLayer(input_size=3, init_rules=2)
    >>> input_data = torch.randn((5, 3))  # Assuming input tensor shape (batch_size, num_input_features)
    >>> weights = torch.ones((5, 2))  # Assuming weight tensor shape (batch_size, num_rules)
    >>> output = consequent_layer(input_data, weights)

    """
    def __init__(self, input_size, dtype=torch.double, init_rules=1, function=f.weighted_linear):
        """
        Initializes a new ConsequentLayer instance.

        Parameters:
        - input_size (int): Number of input features.
        - dtype (torch.dtype): Data type for the premises (default: torch.double).
        - init_rules (int): Number of initial rules.
        - function (callable): Consequent function to apply.

        """
        super(ConsequentLayer, self).__init__()
        self.init_rules = init_rules
        self.input_size = input_size
        self.function = function
        self.consequents = Parameter(2 * torch.rand(init_rules, input_size + 1, dtype=dtype) - 1, requires_grad=True)


    def forward(self, x, w):
        """
        Performs a forward pass to calculate the consequent layer output.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - w (torch.Tensor): Weights tensor.

        Returns:
        - torch.Tensor: Consequent layer output.

        """
        outputs = self.function(x, self.consequents, w)
        return outputs


    @property
    def consequents_structure(self):
        """
        Prints the structure of the consequent parameters.

        """
        print("Consequents Structure:")
        for i in range(self.init_rules):
            print(f"    rule {i + 1} consequent parameters: {self.consequents[i]}")
            
            
            
class OutputLayer(nn.Module):
    """
    Class for representing the last layer (output layer) of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Methods:
    - __init__: Initializes a new OutputLayer instance.
    - forward: Performs a forward pass to calculate the final output.

    Example Usage:
    >>> output_layer = OutputLayer()
    >>> input_data = torch.randn((5, 3))  # Assuming input tensor shape (batch_size, num_input_features)
    >>> output = output_layer(input_data)

    """
    def __init__(self, function=f.sum):
        """
        Initializes a new OutputLayer instance.

        Parameters:
        - function (callable): Aggregation function to apply.

        """
        super(OutputLayer, self).__init__()
        self.function = function

    def forward(self, x):
        """
        Performs a forward pass to calculate the final output.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Final output.

        """
        return self.function(x)
    


