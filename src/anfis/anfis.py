import torch.nn as nn
from torch.nn.parameter import Parameter

import anfis.functions as f
import anfis.layers as l

class Type3ANFIS(nn.Module):
    """
    Class for representing a type 3 Adaptive Neuro-Fuzzy Inference System (ANFIS). Its made up of the following layers:
    
    **Layers:**
    
    .. attribute:: Fuzzification layer (FuzzifyLayer(nn.Module))
        
    .. attribute:: Firing levels layer (FiringLevelsLayer(nn.Module))
        
    .. attribute:: Normalization layer (NormalizationLayer(nn.Module))
    
    .. attribute:: Consequent layer (ConsequentLayer(nn.Module))
    
    .. attribute:: Output layer (OutputLayer(nn.Module))
    
    **Attributes:**
    
    .. attribute:: input_size
    
        The size of the input features.
        
        :type: int
    
    .. attribute:: dtype
    
        The data type of the input data (used to initialize the premises with it)
        
        :type: torch.dtype
    
    .. attribute:: mf_params
    
        List of parameter names for the membership function.
        
        :type: list
        :default: ['mu', 'sigma']
        
    **To initialize it:**
        
    The parameters that must be taken into account are the following:

    :param x_train: Input training data set.
    :type x_train: torch.tensor

    :param init_rules: Number of initial rules (default: 1).
    :type init_rules: int
        
    :param cf: Consequent function to apply (default: weighted_linear).
    :type cf: callable
        
    :param mf: Membership function to apply (default: gaussian2).
    :type mf: callable
        
    :param mf_params: List of membership function parameters (default: ['mu', 'sigma']).
    :type mf_params: list
        
    :param init_mode: Numeric flag for initializing the fuzzy premises (default: 0, meaning it will be initialized based on the input data, otherwise it will be initialized randomly).
    :type init_mode: int
    
    **Example Usage:**
    
    .. code::
    
        >>> input_data = torch.randn((5, 3))  # Assuming input tensor shape (batch_size, num_input_features)
        >>> anfis_model = Type3ANFIS(input_data, init_rules=2)
        >>> output = anfis_model(input_data)
    
    **Methods:**

    """


    def __init__(self, x_train, init_rules=1, cf=f.weighted_linear, mf=f.gaussian2, mf_params=['mu', 'sigma'], init_mode=0):
        """
        To initialize a new Type3ANFIS instance.

        :param x_train: Input training data set.
        :type x_train: torch.tensor

        :param init_rules: Number of initial rules.
        :type init_rules: int
        :default init_rules: 1
        
        :param cf: Consequent function to apply.
        :type cf: callable
        :default cf: weighted_linear
        
        :param mf: Membership function to apply.
        :type mf: callable
        :default mf: gaussian2
        
        :param mf_params: List of membership function parameters.
        :type mf_params: list
        :default mf_params: ['mu', 'sigma']
        
        :param init_mode: Numeric flag for initializing the fuzzy premises (default: 0, meaning it will be initialized based on the input data, otherwise it will be initialized randomly).
        :type init_mode: int
        :deafult init_mode: 0

        """
        super(Type3ANFIS, self).__init__()
        self.input_size = x_train.shape[1]
        self.dtype = x_train.dtype
        self.mf_params = mf_params

        self.fuzzify_layer = l.FuzzifyLayer(x_train, init_rules, mf, mf_params, init_mode)
        self.firing_levels_layer = l.FiringLevelsLayer()
        self.normalization_layer = l.NormalizationLayer()
        self.consequent_layer = l.ConsequentLayer(self.input_size, self.dtype, init_rules, cf)
        self.output_layer = l.OutputLayer()


    def forward(self, x):
        """
        Performs a forward pass through the ANFIS model.
        
        :param x: Input tensor.
        :type x: torch.tensor

        :return: Final output.
        :rtype: torch.tensor

        """
        output = self.fuzzify_layer(x)
        output = self.consequent_layer(x, self.normalization_layer(self.firing_levels_layer(output)))
        output = self.output_layer(output)
        return output


    def intermediate_values(self, x):
        """
        Computes normalized firing levels based on input data.

        :param x: Input tensor.
        :type x: torch.tensor

        :return:
            - w (torch.tensor): Firing levels.
            - w_norm (torch.tensor): Normalized firing levels.
            - outputs (torch.tensor): Outputs by rule of the model

        """
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

        :param premises: New premises.
        :type: torch.tensor

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

        :param consequents: New consequents.
        :type: torch.tensor

        """
        self.consequent_layer.consequents = Parameter(consequents, requires_grad=True)



