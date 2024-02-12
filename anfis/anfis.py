import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import functions as f
import layers as l

class Type3ANFIS(nn.Module):
    """
    Class for representing a type 3 Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Methods:
    - __init__: Initializes a new Type3ANFIS instance.
    - forward: Performs a forward pass through the ANFIS model.
    - firing_levels: Computes firing levels based on input data.
    - normalization_layer: Normalize the firing levels obtained on a previous layer.
    - outputs_by_rule: Computes output for each rule based on input data.
    - premises_structure: Prints the structure of the premises.
    - consequents_structure: Prints the structure of the consequents.
    - set_premises: Sets the premises of the fuzzification layer.
    - set_consequents: Sets the consequents of the consequent layer.

    Example Usage:
    >>> anfis_model = Type3ANFIS(input_size=3, init_rules=2)
    >>> input_data = torch.randn((5, 3))  # Assuming input tensor shape (batch_size, num_input_features)
    >>> output = anfis_model(input_data)

    """


    def __init__(self, input_size, dtype=torch.double, init_rules=1, cf=f.weighted_linear, mf=f.gaussian2, of=sum, mf_params=['mu', 'sigma'], x_train=[]):
        """
        Initializes a new Type3ANFIS instance.

        Parameters:
        - input_size (int): Number of input features.
        - dtype (torch.dtype): Data type for the premises (default: torch.double).
        - init_rules (int): Number of initial rules (default: 1).
        - cf (callable): Consequent function to apply (default: weighted_linear).
        - mf (callable): Membership function to apply (default: gaussian2).
        - of (callable): Aggregation function for the output layer (default: sum).
        - mf_params (list): List of membership function parameters (default: ['mu', 'sigma']).
        - x_train (list): List of training data (default: empty list).

        """
        super(Type3ANFIS, self).__init__()
        self.outputs_by_rule = torch.tensor = torch.tensor([])
        self.input_size = input_size
        self.mf_params = mf_params
        if x_train == []:
            self.fuzzify_layer = l.FuzzifyLayer(input_size, dtype, init_rules, mf, mf_params)
        else:
            self.fuzzify_layer = l.FuzzifyLayer(input_size, dtype, init_rules, mf, mf_params, x_train)
        self.firing_levels_layer = l.FiringLevelsLayer()
        self.normalization_layer = l.NormalizationLayer()
        self.consequent_layer = l.ConsequentLayer(input_size, dtype, init_rules, cf)
        self.output_layer = l.OutputLayer(of)


    def forward(self, x):
        """
        Performs a forward pass through the ANFIS model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Final output.

        """
        output = self.fuzzify_layer(x)
        output = self.consequent_layer(x, self.normalization_layer(self.firing_levels_layer(output), self.rules))
        self.outputs_by_rule = output
        output = self.output_layer(output)
        return output


    def norm_firing_levels(self, m):
        """
        Computes normalized firing levels based on input data.

        Parameters:
        - m (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Firing levels.

        """
        w = self.fuzzify_layer(m)
        w = self.firing_levels_layer(w)
        w = self.normalization_layer(w)
        return w


    @property
    def rules(self):
        """
        Returns the number of rules in the system.

        """
        return self.consequents.shape[0]


    @property
    def premises_structure(self):
        """
        Prints the structure of the premises.

        """
        self.fuzzify_layer.premises_structure


    @property
    def premises(self):
        """
        Return the premises parameters of the fuzzify layer as a torch tensor.

        """
        return self.fuzzify_layer.premises.data


    def set_premises(self, premises):
        """
        Sets the premises of the fuzzification layer.

        Parameters:
        - premises (torch.Tensor): New premises.

        """
        self.fuzzify_layer.premises = Parameter(premises, requires_grad=True)


    @property
    def consequents_structure(self):
        """
        Prints the structure of the consequents.

        """
        self.consequent_layer.consequents_structure


    @property
    def consequents(self):
        """
        Returns the consequents of the consequent layer as a torch.tensor.

        """
        return self.consequent_layer.consequents.data


    def set_consequents(self, consequents):
        """
        Sets the consequents of the consequent layer.

        Parameters:
        - consequents (torch.Tensor): New consequents.

        """
        self.consequent_layer.consequents = Parameter(consequents, requires_grad=True)



