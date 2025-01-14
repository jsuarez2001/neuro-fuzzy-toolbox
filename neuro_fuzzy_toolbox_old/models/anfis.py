import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from neuro_fuzzy_toolbox_old.layers import FuzzifyLayer, FiringLevelsLayer, NormalizationLayer, ConsequentLayer, OutputLayer
from neuro_fuzzy_toolbox_old.func import weighted_linear, gaussian2


class Type3ANFIS(nn.Module):
    """
    Class for representing a type 3 Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Methods:
    - __init__: Initializes a new Type3ANFIS instance.
    - forward: Performs a forward pass through the ANFIS model.
    - intermediate_values: Similar to forward pass but returns the intermediate values obtained by some of the model layers.
    - rules: Returns the number of rules in the system.
    - premises_structure: Prints the structure of the premises.
    - premises: Gets the premises of the fuzzification layer as a tensor.
    - set_premises: Sets the premises parameters of the fuzzification layer.
    - consequents_structure: Prints the structure of the consequents.
    - consequents: Gets the consequents of the consequent layer as a tensor.
    - set_consequents: Sets the consequents parameters of the consequent layer.

    Example Usage:
    >>> input_data = torch.randn((5, 3))  # Assuming input tensor shape (batch_size, num_input_features)
    >>> anfis_model = Type3ANFIS(input_data, init_rules=2)
    >>> output = anfis_model(input_data)

    """


    def __init__(self, x_train, init_rules=1, cf=weighted_linear, mf=gaussian2, mf_params=['mu', 'sigma'], init_mode=0, output_type=None, outputs_amount=1):
        """
        Initializes a new Type3ANFIS instance.

        Parameters:
        - x_train (torch.tensor): input training data set.
        - init_rules (int): Number of initial rules (default: 1).
        - cf (callable): Consequent function to apply (default: weighted_linear).
        - mf (callable): Membership function to apply (default: gaussian2).
        - mf_params (list): List of membership function parameters (default: ['mu', 'sigma']).
        - init_mode (int): Numeric flag for initializing the fuzzy premises (default: 0, meaning it will be initialized based on the input data,
                           otherwise it will be initialized randomly).

        """
        super(Type3ANFIS, self).__init__()
        self.input_size = x_train.shape[1]
        self.dtype = x_train.dtype
        self.mf_params = mf_params

        self.fuzzify_layer = FuzzifyLayer(x_train, init_rules, mf, mf_params, init_mode)
        self.firing_levels_layer = FiringLevelsLayer()
        self.normalization_layer = NormalizationLayer()
        self.consequent_layer = ConsequentLayer(self.input_size, self.dtype, init_rules, cf, outputs_amount)
        self.output_layer = OutputLayer()
        self.last_layer = nn.Identity()
        
        if (output_type == 'binary'):
            self.last_layer = nn.Sigmoid()
        elif (output_type == 'multiclass'):
            self.last_layer = nn.Softmax(dim=0)


    def forward(self, x):
        """
        Performs a forward pass through the ANFIS model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Final output.

        """
        output = self.fuzzify_layer(x)
        output = self.consequent_layer(x, self.normalization_layer(self.firing_levels_layer(output)))
        output = self.output_layer(output)
        output = self.last_layer(output).t()
        return output


    def classes_prediction(self, x):
        """
        Emulates a forward pass throught the ANFIS model, but a thereshold for binary output is applied at the end.

        :param x: Input data.
        :type x: torch.Tensor

        :return: Binary predictions.
        :rtype: torch.Tensor

        """
        with torch.no_grad():
            output = self.forward(x)

        if isinstance(self.last_layer, nn.Softmax):
            output = torch.max(output, dim=1).indices

        elif isinstance(self.last_layer, nn.Sigmoid):
            output = (output >= 0.5).to(int)

        return output


    def intermediate_values(self, x):
        """
        Computes normalized firing levels based on input data.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - w (torch.Tensor): Firing levels.
        - w_norm (torch.Tensor): Normalized firing levels.
        - outputs (torch.Tensor): Outputs by rule of the model

        """
        with torch.no_grad():
          w = self.fuzzify_layer(x)
          w = self.firing_levels_layer(w)
          w_norm = self.normalization_layer(w)
          outputs = self.consequent_layer(x, w_norm)
        return w, w_norm, outputs


    @property
    def rules(self):
        """
        Returns the number of rules in the system.

        """
        return self.consequents.shape[1]


    @property
    def premises_structure(self):
        """
        Prints the structure of the premises.

        """
        return self.fuzzify_layer.premises_structure


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
        return self.consequent_layer.consequents_structure


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