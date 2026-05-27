import torch
import torch.nn as nn
from torch.nn import Parameter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neuro_fuzzy_toolbox.func import GeneralizedBell_MF

class FuzzificationLayer(nn.Module):
    """
    Fuzzification layer for an Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Transforms input data into membership degrees for each MF of each input
    feature. Designed to handle a general ANFIS model where different input
    features may have different numbers of MFs.
    """
    def __init__(self, mf_distribution, membership_function=GeneralizedBell_MF(), features=None, dtype=torch.float32):
        """
        Initializes a new FuzzificationLayer instance.
        
        Args:
            mf_distribution (list[int]): Number of MFs for each input feature.
            membership_function (MembershipFunction): MF type to use. Defaults to ``GeneralizedBell_MF()``.
            features (iterable, optional): Names of the input features as strings, of length ``input_size``. Defaults to ``None``.
            dtype (torch.dtype): Data type for the layer parameters. Defaults to ``torch.float32``.
        """
        super(FuzzificationLayer, self).__init__()
        
        # Input data info
        self._input_size = len(mf_distribution)
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(self._input_size)]
        if features != None and len(features) == self._input_size:
            self.features = features
        
        # Membership function
        self._membership_function = membership_function
        
        # Premise parameters
        self._max_n_mfs = max(mf_distribution)
        self._mf_distribution = torch.tensor(mf_distribution)
        
        premises = nn.ParameterList()
        for n_mfs in self._mf_distribution:
           premises.append(Parameter(self._membership_function.random_single_feature_mfs(n_mfs, dtype), requires_grad=True))
        self._premises = premises
        
    
    def forward(self, x):
        """
        Forward pass of the fuzzification layer.

        Computes the membership degrees for each input feature.

        Args:
            x (torch.Tensor): Input data of shape ``(batch_size, input_size)``.

        Returns:
            torch.Tensor: Membership degrees of shape ``(batch_size, input_size, max_n_mfs)``, where ``max_n_mfs`` is the maximum number of MFs across all input features.

        Note:
            Features with fewer MFs than ``max_n_mfs`` are zero-padded to
            match the output shape. This padding is applied during the forward
            pass and may be improved in future versions by embedding it
            directly into the premise parameter tensors at instantiation time.
        """
        reshaped_premises = torch.tensor([], dtype=self._dtype)
                
        # Reshape premises
        for mf in self._premises:
            if mf.size(0) < self._max_n_mfs:
                mf = torch.cat((mf, torch.zeros(self._max_n_mfs - mf.size(0), mf.size(1), dtype=self._dtype)), 0)
            reshaped_premises = torch.cat((reshaped_premises, mf.unsqueeze(dim=0)), 0)
            
        return self._membership_function(x, reshaped_premises)
    
    
    def init_premises(self, x_train):
        """
        Initializes the MF premise parameters from training data.

        Args:
            x_train (torch.Tensor): Training input data of shape ``(n_samples, input_size)``.
        """
        self._dtype = x_train.dtype
        self._premises = nn.ParameterList(self._membership_function.general_initialize_premises(x_train, self._mf_distribution))
        self._membership_function._max_val_plot = x_train.max().item()
        self._membership_function._min_val_plot = x_train.min().item()
    
    def get_premises(self):
        """
        Returns the current premise parameters.

        Returns:
            list[torch.Tensor]: List of premise parameter tensors, one per input feature.
        """
        return [mf.data.clone().detach() for mf in self._premises]
    
    
    def set_premises(self, premises):
        """
        Sets the premise parameters for all input features.

        Args:
            premises (list[torch.Tensor]): List of premise parameter tensors. Each tensor must have shape ``(num_mfs, mf_params)``, where ``mf_params`` is the number of parameters of the MF type in use.
        """
        for i, premise in enumerate(premises):
            self._premises[i] = Parameter(premise, requires_grad=True)
            
    
    def get_premises_as_parameters_list(self):
        """
        Returns the premise parameters as a PyTorch ParameterList.

        Returns:
            nn.ParameterList: Premise parameters wrapped as a list of trainable parameters.
        """
        return self._premises
    
    
    @property
    def num_mfs(self):
        """
        Number of MFs per input feature.

        Returns:
            torch.Tensor: Tensor containing the number of MFs for each input feature.
        """
        return self._mf_distribution
    
    
    @property
    def get_premises_structure(self):
        """
        Structure of the premise parameters.

        Returns:
            pd.DataFrame: DataFrame describing the premise parameters for each input feature and MF.
        """
        
        columns = pd.MultiIndex.from_product(
            [self.features, self._membership_function._params]
        )

        df = pd.DataFrame(columns=columns)

        mfs = [f'MF {i + 1}' for i in range(self._max_n_mfs)]

        for i in range(self._input_size):
            num_mfs = self._mf_distribution[i]
            for param_name in self._membership_function._params:
                column_data = [self._premises[i][mf_idx, self._membership_function._params.index(param_name)].item() 
                               if mf_idx < num_mfs else None 
                               for mf_idx in range(self._max_n_mfs)]

                df[(self.features[i], param_name)] = pd.Series(column_data, index=mfs)

        return df


    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False, linestyles='-', linewidths=2.5):
        """
        Plots the premise membership functions.

        Args:
            mf (int, optional): Index of the MF to plot. If ``None``, all MFs are plotted. Defaults to ``None``.
            input_dim (int, optional): Index of the input feature to plot. If ``None``, all input features are plotted. Defaults to ``None``.
            group_by_dim (bool): If ``True``, all MFs for each input feature are grouped into a single plot. Defaults to ``False``.
            linestyles (str or list[str]): Line style or list of line styles to cycle through when plotting MFs. Accepted values are ``'-'``, ``'--'``, ``'-.'``, and ``':'``. Defaults to ``'-'``.
            linewidths (float or list[float]): Line width or list of line widths to cycle through when plotting MFs. Defaults to ``2.5``.
        """
        variables = self.features
        dataframe = self.get_premises_structure
        
        x = np.linspace(self._membership_function._min_val_plot, self._membership_function._max_val_plot, 500)
        
        # Determine which mfs and dimensions to plot
        if mf is not None:
            mf = f'{mf}'
            # Validate that the mf exists
            if mf not in dataframe.index:
                raise ValueError(f"'{mf}'index not found in premises. Available mfs: {dataframe.index.tolist()}")
            mfs_to_plot = [mf]
        else:
            mfs_to_plot = dataframe.index
            
        # Validate input dimension
        if input_dim is not None:
            if not isinstance(input_dim, int) or input_dim < 0 or input_dim >= len(variables):
                raise ValueError(f"input_dim must be between 0 and {len(variables)-1}")
            dims_to_plot = [input_dim]
        else:
            dims_to_plot = range(len(variables))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(mfs_to_plot)))
            
        if group_by_dim:
            fig, axes = plt.subplots(nrows=len(dims_to_plot), ncols=1, figsize=(10, 5*len(dims_to_plot)), squeeze=False)
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                ax = axes[j, 0]
                for mf_idx, mf in enumerate(mfs_to_plot):
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            color = colors[mf_idx % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[mf_idx % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, label=f'{mf}', color=color, linestyle=linestyle, linewidth=linewidth)
                    except KeyError:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
                ax.set_title(f'Membership Functions for {var}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('x', fontsize=11)
                ax.set_ylabel('Membership Value', fontsize=11)
                ax.legend(fontsize=10, loc='best')
                ax.set_ylim([0, 1.1])

        else:
            # Calculate subplot dimensions
            n_mfs = len(mfs_to_plot)
            n_dims = len(dims_to_plot)

            # Create subplots based on the number of mfs and dimensions
            if n_mfs == 1 and n_dims == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = np.array([[ax]])
            else:
                fig, axes = plt.subplots(nrows=n_dims, ncols=n_mfs, figsize=(8*n_mfs, 6*n_dims), squeeze=False)

            for i, mf in enumerate(mfs_to_plot):
                for j, dim in enumerate(dims_to_plot):
                    var = variables[dim]
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax = axes[j, i] if n_mfs > 1 else axes[i, j]
                            color = colors[i % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[i % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{mf}, {var}')
                            ax.set_title(f'{mf}, {var}', fontsize=11, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim([0, 1.1])
                            if (i == n_mfs - 1) or (j == n_dims - 1):
                                ax.set_xlabel('x', fontsize=10)
                            if j == 0 or i == 0:
                                ax.set_ylabel('Membership Value', fontsize=10)
                    except KeyError as e:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
            plt.tight_layout()
            plt.show()



class h_FuzzificationLayer(nn.Module):
    """
    Homogeneous fuzzification layer for an Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Transforms input data into membership degrees for each MF of each input
    feature. Unlike :class:`FuzzificationLayer`, this layer enforces the same
    number of MFs for every input feature, constraining each to the same
    number of linguistic variables.
    """
    def __init__(self, input_size, num_mfs=1, membership_function=GeneralizedBell_MF(), features=None, dtype=torch.float32):
        """
        Initializes a new h_FuzzificationLayer instance.

        Args:
            input_size (int): Number of input features.
            num_mfs (int): Number of MFs per input feature. Defaults to ``1``.
            membership_function (MembershipFunction): MF type to use. Defaults to ``GeneralizedBell_MF()``.
            features (iterable, optional): Names of the input features as strings, of length ``input_size``. Defaults to ``None``.
            dtype (torch.dtype): Data type for the layer parameters. Defaults to ``torch.float32``.
        """
        super(h_FuzzificationLayer, self).__init__()

        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(input_size)]
        if features != None and len(features) == input_size:
            self.features = features
        
        # Membership function
        self._membership_function = membership_function

        # Initialize premise parameters
        self._premises = Parameter(self._membership_function.random_premises(input_size, num_mfs, dtype), requires_grad=True)


    def forward(self, x):
        """
        Forward pass of the homogeneous fuzzification layer.

        Computes the membership degrees for each input feature.

        Args:
            x (torch.Tensor): Input data of shape ``(batch_size, input_size)``.

        Returns:
            torch.Tensor: Membership degrees of shape ``(batch_size, input_size, num_mfs)``.
        """
        return self._membership_function(x, self._premises)
    
    
    def init_premises(self, x_train):
        """
        Initializes the MF premise parameters from training data.
        
        Args:
            x_train (torch.Tensor): Training input data of shape
                ``(n_samples, input_size)``.
        """
        self._dtype = x_train.dtype
        self._premises = Parameter(self._membership_function.initialize_premises(x_train=x_train, num_mfs=self._premises.data.shape[1]), requires_grad=True)
        self._membership_function._max_val_plot = x_train.max().item()
        self._membership_function._min_val_plot = x_train.min().item()

    def get_premises(self):
        """
        Returns the current premise parameters.

        Returns:
            torch.Tensor: Premise parameter tensor of shape ``(input_size, num_mfs, mf_params)``, where ``mf_params`` is the number of parameters of the MF type in use.
        """
        return self._premises.data.clone().detach()
    
    
    def set_premises(self, premises):
        """
        Sets the premise parameters for all input features.

        Args:
            premises (torch.Tensor): Premise parameter tensor of shape ``(input_size, num_mfs, mf_params)``, where ``mf_params`` is the number of parameters of the MF type in use.
        """
        self._premises = Parameter(premises, requires_grad=True)
        
        
    def get_premises_as_parameters_list(self):
        """
        Returns the premise parameters as a single-element list of trainable parameters, useful for passing to optimizers.

        Returns:
            list[nn.Parameter]: List containing a single ``nn.Parameter`` with all premise parameters.
        """
        return [self._premises]
    
    
    @property
    def num_mfs(self):
        """
        Number of MFs per input feature.

        Returns:
            int: Number of MFs assigned to each input feature.
        """
        return self.get_premises().shape[1]
    

    @property
    def get_premises_structure(self):
        """
        Structure of the premise parameters.

        Returns:
            pd.DataFrame: DataFrame describing the premise parameters for each input feature and MF.
        """
        premises = self.get_premises()
        
        mfs = [f"rule {i + 1}" for i in range(premises.shape[1])]

        columns = pd.MultiIndex.from_product(
            (self.features, self._membership_function._params),
        )

        df = pd.DataFrame(index=mfs, columns=columns)

        for i in range(self._input_size):
            for j, param in enumerate(self._membership_function._params):
                column_data = premises[i, :, j].tolist()
                df[(self.features[i], param)] = pd.Series(column_data, index=mfs)

        return df
    
    
    
    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False, linestyles='-', linewidths=2.5):
        """
        Plots the premise membership functions.

        Args:
            mf (int, optional): Index of the MF to plot. If ``None``, all MFs are plotted. Defaults to ``None``.
            input_dim (int, optional): Index of the input feature to plot. If ``None``, all input features are plotted. Defaults to ``None``.
            group_by_dim (bool): If ``True``, all MFs for each input feature are grouped into a single plot. Defaults to ``False``.
            linestyles (str or list[str]): Line style or list of line styles to cycle through when plotting MFs. Accepted values are ``'-'``, ``'--'``, ``'-.'``, and ``':'``. Defaults to ``'-'``.
            linewidths (float or list[float]): Line width or list of line widths to cycle through when plotting MFs. Defaults to ``2.5``.
        """
        variables = [f'{self.features[i]}' for i in range(self._input_size)]
        dataframe = self.get_premises_structure

        x = np.linspace(self._membership_function._min_val_plot, self._membership_function._max_val_plot, 500)

        # Determine which mfs and dimensions to plot
        if mf is not None:
            mf = f'{mf}'
            # Validate that the mf exists
            if mf not in dataframe.index:
                raise ValueError(f"'{mf}' index not found in premises. Available mfs: {dataframe.index.tolist()}")
            mfs_to_plot = [mf]
        else:
            mfs_to_plot = dataframe.index

        # Validate input dimension
        if input_dim is not None:
            if not isinstance(input_dim, int) or input_dim < 0 or input_dim >= len(variables):
                raise ValueError(f"input_dim must be between 0 and {len(variables)-1}")
            dims_to_plot = [input_dim]
        else:
            dims_to_plot = range(len(variables))
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(mfs_to_plot)))
            
        if group_by_dim:
            fig, axes = plt.subplots(nrows=len(dims_to_plot), ncols=1, figsize=(10, 5*len(dims_to_plot)), squeeze=False)
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                ax = axes[j, 0]
                for mf_idx, mf in enumerate(mfs_to_plot):
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            #value = dataframe.loc[mf, f'{param} ({var})']
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            color = colors[mf_idx % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[mf_idx % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, label=f'{mf}', color=color, linestyle=linestyle, linewidth=linewidth)
                    except KeyError:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
                ax.set_title(f'Membership Functions for {var}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('x', fontsize=11)
                ax.set_ylabel('Membership Value', fontsize=11)
                ax.legend(fontsize=10, loc='best')
                ax.set_ylim([0, 1.1])

        else:
            # Calculate subplot dimensions
            n_mfs = len(mfs_to_plot)
            n_dims = len(dims_to_plot)

            # Create subplots based on the number of mfs and dimensions
            if n_mfs == 1 and n_dims == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = np.array([[ax]])
            else:
                fig, axes = plt.subplots(nrows=n_dims, ncols=n_mfs, figsize=(8*n_mfs, 6*n_dims), squeeze=False)

            for i, mf in enumerate(mfs_to_plot):
                for j, dim in enumerate(dims_to_plot):
                    var = variables[dim]
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax = axes[j, i] if n_mfs > 1 else axes[i, j]
                            color = colors[i % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[i % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{mf}, {var}')
                            ax.set_title(f'{mf}, {var}', fontsize=11, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim([0, 1.1])
                            if (i == n_mfs - 1) or (j == n_dims - 1):
                                ax.set_xlabel('x', fontsize=10)
                            if j == 0 or i == 0:
                                ax.set_ylabel('Membership Value', fontsize=10)
                    except KeyError as e:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
            plt.tight_layout()
            plt.show()
            
            
    
class rule_reduced_FuzzificationLayer(nn.Module):
    """
    Fuzzification layer for a rule-reduced Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Transforms input data into membership degrees for each MF of each input
    feature. Like :class:`h_FuzzificationLayer`, this layer enforces the same
    number of MFs for every input feature. It is specifically designed for the
    rule-reduced ANFIS model, which supports structural adaptation via the
    SONFIS algorithm.
    """
    def __init__(self, input_size, num_mfs=1, membership_function=GeneralizedBell_MF(), features=None, dtype=torch.float32):
        """
        Initializes a new rule_reduced_FuzzificationLayer instance.

        Args:
            input_size (int): Number of input features.
            num_mfs (int): Number of MFs per input feature. Defaults to ``1``.
            membership_function (MembershipFunction): MF type to use. Defaults to ``GeneralizedBell_MF()``.
            features (iterable, optional): Names of the input features as strings, of length ``input_size``. Defaults to ``None``.
            dtype (torch.dtype): Data type for the layer parameters. Defaults to ``torch.float32``.
        """
        super(rule_reduced_FuzzificationLayer, self).__init__()

        # Input data info
        self._input_size = input_size
        self._dtype = dtype
        self.features = [f"x{i}" for i in range(input_size)]
        if features != None and len(features) == input_size:
            self.features = features
        
        # Membership function
        self._membership_function = membership_function

        # Initialize premise parameters
        self._premises = nn.ParameterList([
            nn.Parameter(premise, requires_grad=True) for premise in self._membership_function.random_premises(
                input_size=input_size, 
                num_mfs=num_mfs, 
                dtype=dtype
            ).unbind(1)
        ])


    def forward(self, x):
        """
        Forward pass of the rule-reduced fuzzification layer.

        Computes the membership degrees for each input feature.

        Args:
            x (torch.Tensor): Input data of shape ``(batch_size, input_size)``.

        Returns:
            torch.Tensor: Membership degrees of shape ``(batch_size, input_size, num_mfs)``.
        """
        return self._membership_function(x, torch.stack([premise for premise in self._premises], 1))


    def init_premises(self, x_train):
        """
        Initializes the MF premise parameters from training data.

        Args:
            x_train (torch.Tensor): Training input data of shape ``(n_samples, input_size)``.
        """
        self._dtype = x_train.dtype
        self._premises = nn.ParameterList([
            nn.Parameter(premise, requires_grad=True) for premise in self._membership_function.initialize_premises(x_train=x_train, num_mfs=len(self._premises)).unbind(1)
        ])
        self._membership_function._max_val_plot = x_train.max().item()
        self._membership_function._min_val_plot = x_train.min().item()
        
        
    def get_premises(self):
        """
        Returns the current premise parameters.

        Returns:
            torch.Tensor: Premise parameter tensor of shape ``(input_size, num_mfs, mf_params)``, where ``mf_params`` is the number of parameters of the MF type in use.
        """
        return torch.stack([premise.data.clone().detach() for premise in self._premises], 1)
        
        
    def set_premises(self, premises):
        """
        Sets the premise parameters for all input features.

        Args:
            premises (torch.Tensor): Premise parameter tensor of shape ``(input_size, num_mfs, mf_params)``, where ``mf_params`` is the number of parameters of the MF type in use.
        """
        self._premises = nn.ParameterList([
            nn.Parameter(premise, requires_grad=True) for premise in premises.unbind(1)
        ])
        
        
    def get_premises_as_parameters_list(self):
        """
        Returns the premise parameters as a PyTorch ParameterList, useful
        for passing to optimizers.

        Returns:
            nn.ParameterList: Premise parameters wrapped as a list of trainable parameters.
        """
        return self._premises
    
    
    @property
    def num_mfs(self):
        """
        Number of MFs per input feature.

        Returns:
            int: Number of MFs assigned to each input feature.
        """
        return self.get_premises().shape[1]
        
    
    @property
    def get_premises_structure(self):
        """
        Structure of the premise parameters.

        Returns:
            pd.DataFrame: DataFrame describing the premise parameters for
            each input feature and MF.
        """
        premises_tensor = torch.stack([premise.data.clone().detach() for premise in self._premises], 1)
        n_mfs = premises_tensor.shape[1]

        mfs = [f"rule {i + 1}" for i in range(n_mfs)]
        
        multi_columns = pd.MultiIndex.from_product(
            (self.features, self._membership_function._params)
        )

        df = pd.DataFrame(index=mfs, columns=multi_columns)

        for i in range(self._input_size):
            for j, param in enumerate(self._membership_function._params):
                column_data = premises_tensor[i, :, j].tolist()
                df.loc[:, (self.features[i], param)] = pd.Series(column_data, index=mfs)

        return df
    
    
    def plot_premises(self, mf=None, input_dim=None, group_by_dim=False, linestyles='-', linewidths=2.5):
        """
        Plots the premise membership functions.

        Args:
            mf (int, optional): Index of the MF to plot. If ``None``, all MFs are plotted. Defaults to ``None``.
            input_dim (int, optional): Index of the input feature to plot. If ``None``, all input features are plotted. Defaults to ``None``.
            group_by_dim (bool): If ``True``, all MFs for each input feature are grouped into a single plot. Defaults to ``False``.
            linestyles (str or list[str]): Line style or list of line styles to cycle through when plotting MFs. Accepted values are ``'-'``, ``'--'``, ``'-.'``, and ``':'``. Defaults to ``'-'``.
            linewidths (float or list[float]): Line width or list of line widths to cycle through when plotting MFs. Defaults to ``2.5``.
        """
        variables = [f'{self.features[i]}' for i in range(self._input_size)]
        dataframe = self.get_premises_structure

        x = np.linspace(self._membership_function._min_val_plot, self._membership_function._max_val_plot, 500)

        # Determine which mfs and dimensions to plot
        if mf is not None:
            mf = f'{mf}'
            # Validate that the mf exists
            if mf not in dataframe.index:
                raise ValueError(f"'{mf}' index not found in premises. Available mfs: {dataframe.index.tolist()}")
            mfs_to_plot = [mf]
        else:
            mfs_to_plot = dataframe.index

        # Validate input dimension
        if input_dim is not None:
            if not isinstance(input_dim, int) or input_dim < 0 or input_dim >= len(variables):
                raise ValueError(f"input_dim must be between 0 and {len(variables)-1}")
            dims_to_plot = [input_dim]
        else:
            dims_to_plot = range(len(variables))
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(mfs_to_plot)))
            
        if group_by_dim:
            fig, axes = plt.subplots(nrows=len(dims_to_plot), ncols=1, figsize=(10, 5*len(dims_to_plot)), squeeze=False)
            for j, dim in enumerate(dims_to_plot):
                var = variables[dim]
                ax = axes[j, 0]
                for mf_idx, mf in enumerate(mfs_to_plot):
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            #value = dataframe.loc[mf, f'{param} ({var})']
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            color = colors[mf_idx % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[mf_idx % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, label=f'{mf}', color=color, linestyle=linestyle, linewidth=linewidth)
                    except KeyError:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
                ax.set_title(f'Membership Functions for {var}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('x', fontsize=11)
                ax.set_ylabel('Membership Value', fontsize=11)
                ax.legend(fontsize=10, loc='best')
                ax.set_ylim([0, 1.1])

        else:
            # Calculate subplot dimensions
            n_mfs = len(mfs_to_plot)
            n_dims = len(dims_to_plot)

            # Create subplots based on the number of mfs and dimensions
            if n_mfs == 1 and n_dims == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = np.array([[ax]])
            else:
                fig, axes = plt.subplots(nrows=n_dims, ncols=n_mfs, figsize=(8*n_mfs, 6*n_dims), squeeze=False)

            for i, mf in enumerate(mfs_to_plot):
                for j, dim in enumerate(dims_to_plot):
                    var = variables[dim]
                    try:
                        params = []
                        for param in self._membership_function._params:
                            value = dataframe.loc[mf, (var, param)]
                            if pd.isna(value):
                                break
                            params.append(value)
                        else:
                            y = self._membership_function._simple_numpy_implementation(x, *params)
                            ax = axes[j, i] if n_mfs > 1 else axes[i, j]
                            color = colors[i % len(colors)]
                            if linestyles != '-' and linestyles != '--' and linestyles != '-.'  and linestyles != ':':
                                linestyle = linestyles[i % len(linestyles)]
                            else:
                                linestyle = linestyles
                            if isinstance(linewidths, list):
                                linewidth = linewidths[mf_idx % len(linewidths)]
                            else:
                                linewidth = linewidths
                            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{mf}, {var}')
                            ax.set_title(f'{mf}, {var}', fontsize=11, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim([0, 1.1])
                            if (i == n_mfs - 1) or (j == n_dims - 1):
                                ax.set_xlabel('x', fontsize=10)
                            if j == 0 or i == 0:
                                ax.set_ylabel('Membership Value', fontsize=10)
                    except KeyError as e:
                        print(f"Warning: Could not find parameters for membership function '{mf}' and variable '{var}'")
                        continue
                    
            plt.tight_layout()
            plt.show()