import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import functions as f

class FuzzifyLayer(nn.Module):
    '''
    Fuzzification layer of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Attributes:
    - input_size (int): The size of the input features.
    - dtype (torch.dtype): The data type of the input data (used to initialize the premises with it)
    - mf (function): Membership function to use (default: gaussian2).
    - params (list): List of parameter names for the membership function (default: ['mu', 'sigma']).
    - premises (torch.nn.Parameter): Trainable parameters for the fuzzification layer.

    Methods:
    - __init__: Initializes a new FuzzifyLayer instance.
    - init_premises: Initializes the premises based on input training data.
    - forward: Performs a forward pass through the fuzzification layer.
    - premises_structure: Prints the structure of the fuzzy premises.

    Example Usage:
    >>> input_data = torch.randn((3, 4))
    >>> fuzzify_layer = FuzzifyLayer(input_data, init_rules=3)
    >>> membership_values = fuzzify_layer(input_data)

    '''
    def __init__(self, x_train, init_rules=1, mf=f.gaussian2, params=['mu', 'sigma'], init_mode=0):
        """
        Initializes a new FuzzifyLayer instance.

        Parameters:
        - x_train (torch.tensor): Input training data.
        - init_rules (int): The number of initial fuzzy rules (default: 1).
        - mf (function): Membership function to use (default: gaussian2).
        - params (list): List of parameter names for the membership function (default: ['mu', 'sigma']).
        - init_mode (int): Numeric flag for initializing the fuzzy premises (default: 0, meaning it will be initialized based on the input data,
                           otherwise it will be initialized randomly).

        """
        super(FuzzifyLayer, self).__init__()
        self.input_size = x_train.shape[1]
        self.dtype = x_train.dtype
        self.mf = mf
        self.params = params

        # Initialize premises
        if init_mode != 0: #random
            prems = 2 * torch.rand(self.input_size, init_rules, len(params), dtype=self.dtype) - 1
            prems[:,:,1] = (prems[:,:,1] + 1)/2
            self.premises = Parameter(prems, requires_grad=True)
        else: #based on the training input data
            self.premises = Parameter(self.init_premises(x_train, init_rules), requires_grad=True)


    def init_premises(self, x_train, init_rules):
        """
        Initializes the fuzzy premises based on input training data.

        Parameters:
        - x_train (torch.Tensor): Training data for initializing fuzzy premises.
        - init_rules (int): The number of initial fuzzy rules.

        Returns:
        - torch.Tensor: Initialized fuzzy premises.

        """
        premises = torch.zeros(self.input_size, init_rules, len(self.params), dtype=x_train.dtype)
        if init_rules > 1:
            min = torch.min(x_train, dim=0).values
            max = torch.max(x_train, dim=0).values
            stp = (max - min) / (init_rules - 1)
            for i in range(self.input_size):
                h = torch.arange(min[i], max[i] + stp[i], stp[i])
                premises[i, :, 0] = h[:init_rules]
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
        for i in range(self.premises.data.shape[1]):
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


class NormalizationLayer(nn.Module):
    """
    Class for normalize the firing levels in an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Methods:
    - __init__: Initializes a new NormalizeLayer instance.
    - forward: Performs a forward pass to normalize the firing levels obtained on a previous layer.

    Example Usage:
    >>> normalization_layer = NormalizationLayer()
    >>> norm_firing_levels = normalization_layer(firing_levels) #assuming 'firing_levels' is the tensor obtained from the Firing Levels Layer

    """
    def __init__(self):
        """
        Initializes a new FiringLevelsLayer instance.

        """
        super(NormalizationLayer, self).__init__()


    def forward(self, w):
        """
        Performs a forward pass through the layer to normalize the firing levels.

        Parameters:
        - x (torch.Tensor): Input tensor containing the membership values for each rule.

        Returns:
        - torch.Tensor: Output tensor (Normalized Firing levels).

        """
        sum = torch.sum(w, dim=1, keepdim=True)
        sum[sum == 0] = 1
        w = w/sum
        return w


class ConsequentLayer(nn.Module):
    """
    Class for representing the fourth layer (consequent layer) of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.
    
    Attributes:
    - function (function): Consequent function to use (default: weighted_linear).
    - consequents (torch.nn.Parameter): Trainable parameters for the consequent layer.


    Methods:
    - __init__: Initializes a new ConsequentLayer instance.
    - forward: Performs a forward pass to calculate the consequent layer output.
    - consequents_structure: Prints the structure of the consequent parameters.

    Example Usage:
    >>> input_data = torch.randn((5, 3))  # Assuming input tensor shape (batch_size, num_input_features)
    >>> consequent_layer = ConsequentLayer(input_data.shape[1], input_data.dtype, init_rules=2)
    >>> output = consequent_layer(input_data, weights) # Assuming weight is the tensor obtained from the normalization layer with shape (batch_size, num_rules)

    """
    def __init__(self, input_size, dtype, init_rules=1, function=f.weighted_linear):
        """
        Initializes a new ConsequentLayer instance.

        Parameters:
        - input_size (int): Number of input features.
        - dtype (torch.dtype): Data type for the consequents.
        - init_rules (int): Number of initial rules.
        - function (callable): Consequent function to apply.

        """
        super(ConsequentLayer, self).__init__()
        self.function = function
        self.consequents = Parameter(2 * torch.rand(init_rules, input_size + 1, dtype=dtype) - 1, requires_grad=True)


    def forward(self, x, w):
        """
        Performs a forward pass to calculate the consequent layer output.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - w (torch.Tensor): Weights tensor.

        Returns:
        - torch.Tensor: Output tensor (Outputs by rule of the ANFIS model).

        """
        outputs = self.function(x, self.consequents, w)
        return outputs


    @property
    def consequents_structure(self):
        """
        Prints the structure of the consequent parameters.

        """
        print("Consequents Structure:")
        for i in range(self.consequents.data.shape[0]):
            print(f"    rule {i + 1} consequent parameters: {self.consequents[i]}")
            
            
            
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


