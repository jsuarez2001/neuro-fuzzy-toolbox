import torch
import torch.nn as nn
import numpy as np

from abc import abstractmethod

class MembershipFunction(nn.Module):
    """
    Abstract base class for membership functions.
    
    Defines the interface required for implementing membership functions within the toolbox. 
    Intended as a guide for future implementations.
    """
    def __init__(self):
        super(MembershipFunction, self).__init__()
        self._params = None
        
        self._max_val_plot = None # For plotting
        self._min_val_plot = None # For plotting
        
    @abstractmethod
    def _simple_numpy_implementation(self, x, *args): # Simple numpy implementation needed for plotting
        """
        NumPy implementation of the membership function, used for plotting.
        
        Args:
            x (numpy.ndarray): Input array.
            *args: MF parameters.
        """
        pass
    
    @abstractmethod
    def _simple_alpha_cut(self, alpha, *args):
        """
        Computes the alpha-cut interval of the membership function.

        Args:
            alpha (float): Membership degree threshold in the range ``[0, 1]``.
            *args: MF parameters.

        Returns:
            numpy.ndarray: Array of 2 elements representing the interval over which the membership degree is greater than ``alpha``.
        """
        pass
    
    @abstractmethod
    def _get_center(self, *args):
        """
        Returns the center of the membership function, used for plotting.
                
        Args:
            *args: MF parameters.

        Returns:
            float: Center of the membership function.
        """
        pass
        
    @abstractmethod
    def forward(self, x, premises):
        """
        Forward pass of the membership function.
        
        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_size)``.
            premises (torch.tensor): Premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
            
        Returns:
            torch.tensor: Membership degrees of shape ``(batch_size, input_size, num_mfs)``.
        """
        pass
    
    @abstractmethod
    def initialize_premises(self, x_train, num_mfs):
        """
        Initializes the premise parameters from training data, assuming the same number of MFs for every input feature.
        
        Args:
            x_train (torch.tensor): Training input data of shape ``(n_samples, input_size)``.
            num_mfs (int): Number of MFs per input feature.
            
        Returns:
            torch.tensor: Initialized premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        """
        pass
    
    @abstractmethod
    def general_initialize_premises(self, x_train, mf_distribution):
        """
        Initializes the premise parameters from training data, allowing a different number of MFs per input feature.
        
        Args:
            x_train (torch.tensor): Training input data of shape ``(n_samples, input_size)``.
            mf_distribution (list[int]): Number of MFs for each input feature.
            
        Returns:
            list[torch.Tensor]: List of premise parameter tensors, one per input feature. The tensor at index ``i`` has shape 
            ``(input_size, mf_distribution[i], len(self._params))``.
        """
        pass
    
    @abstractmethod
    def random_premises(self, input_size, num_mfs, dtype):
        """
        Generates randomly initialized premise parameters in the range ``[-1, 1]``, with certain parameters constrained to 
        positive values depending on the MF type.
        
        Args:
            input_size (int): Number of input features.
            num_mfs (int): Number of MFs per input feature.
            dtype (torch.dtype): Data type of the premise parameters.
            
        Returns:
            torch.tensor: Randomly initialized premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        """
        pass
    
    @abstractmethod
    def random_single_feature_mfs(self):
        """
        Generates randomly initialized premise parameters for a single input feature in the range ``[-1, 1]``, 
        with certain parameters constrained to positive values depending on the MF type.
        
        Args:
            n_mfs (int): Number of MFs.
            dtype (torch.dtype): Data type of the premise parameters.
            
        Returns:
            torch.tensor: Randomly initialized premise parameters of shape ``(n_mfs, len(self._params))``.
        """
        pass
    
    @abstractmethod
    def _grow_new_premise_parameters(self, means, stds):
        """
        Generates new premise parameters from provided means and standard
        deviations, used during structural adaptation in the SONFIS algorithm
        (grow operation). Assumes the same number of MFs for every input feature.
        
        Args:
            means (torch.tensor): Per-feature MF centers of shape ``(num_new_mfs, input_size)``, where ``num_new_mfs`` is the
                number of new MFs to add to each input feature.
            stds (torch.tensor): Per-feature MF spreads of shape ``(num_new_mfs, input_size)``, matching the shape of ``means``.
            
        Returns:
            torch.tensor: New premise parameters of shape ``(input_size, num_new_mfs, len(self._params))``.
        """
        pass
    
    @abstractmethod
    def _split_premise_parameters(self, premises):
        """
        Splits each MF in the given premises into two, used during structural
        adaptation in the SONFIS algorithm (split operation). Assumes the same
        number of MFs for every input feature.
        
        Args:
            premises (torch.tensor): Premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
            
        Returns:
            torch.tensor: Resultant premise parameters of shape ``(input_size, 2 * num_mfs, len(self._params))``, where each original MF has been replaced by two new MFs.
        """
        pass


class Gaussian_MF(MembershipFunction):
    """  
    Gaussian membership function, defined as: 

    .. math::

        gaussian(x) = e^{-\\frac{(x - \\mu)^2}{2\\sigma^2}}

    donde:
        - :math:`x` is the input variable.
        - :math:`\\mu` is the center of the function.
        - :math:`\\sigma` is the standard deviation, controlling the width of the curve.
    """
    def __init__(self):
        super(Gaussian_MF, self).__init__()
        self._params = ["mu", "sigma"]
        
        self._min_val_plot = -2
        self._max_val_plot = 2
        
    def _simple_numpy_implementation(self, x, mu, sigma):
        """
        NumPy implementation of the Gaussian MF, used for plotting.
        
        Args:
            x (numpy.ndarray): Input array.
            mu (float): Center of the Gaussian MF.
            sigma (float): Standard deviation of the Gaussian MF.
            
        Returns:
            numpy.ndarray: Membership degrees for each value in ``x``.
        """
        return np.exp(-0.5 * np.power((x - mu)/sigma, 2))
    
    def _simple_alpha_cut(self, alpha, mu, sigma):
        """
        Computes the alpha-cut interval of the Gaussian MF.

        Args:
            alpha (float): Membership degree threshold in the range ``[0, 1]``.
            mu (float): Center of the Gaussian MF.
            sigma (float): Standard deviation of the Gaussian MF.

        Returns:
            np.array: Array of 2 elements representing the interval over which the membership degree is greater than ``alpha``.
        """
        sigma = abs(sigma)
        r = np.sqrt(-2*np.log(alpha))
        L = mu - sigma*r
        U = mu + sigma*r
        return np.array([L, U])
    
    def _get_center(self, mu, sigma):
        """
        Returns the center of the Gaussian MF, used for plotting.
        
        Args:
            mu (float): Center of the Gaussian MF.
            sigma (float): Standard deviation of the Gaussian MF.
            
        Returns:
            float: Center of the membership function.
        """
        return mu

    def forward(self, x, premises):
        """
        Forward pass of the Gaussian membership function.
        
        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_size)``.
            premises (torch.Tensor): Premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        
        Returns:
            torch.Tensor: Membership degrees of shape ``(batch_size, input_size, num_mfs)``.
        """
        return torch.exp(-0.5 * torch.pow((x.unsqueeze(x.dim()) - premises[:, :, 0])/torch.where(premises[:, :, 1] == 0, torch.tensor(1e-6), premises[:, :, 1]), 2))

    def initialize_premises(self, x_train, num_mfs):
        """
        Initializes the Gaussian MF premise parameters from training data, assuming the same number of MFs for every input feature.
        
        Args:
            x_train (torch.Tensor): Training input data of shape ``(n_samples, input_size)``.
            num_mfs (int): Number of MFs per input feature.
            
        Returns:
            torch.Tensor: Initialized premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        """
        input_size = x_train.shape[1]
        premises = torch.zeros(input_size, num_mfs, len(self._params), dtype=x_train.dtype)
        
        if num_mfs > 1:
            min_val = torch.min(x_train, dim=0).values
            max_val = torch.max(x_train, dim=0).values
            stp = (max_val - min_val) / (num_mfs - 1)
            for i in range(input_size):
                h = torch.arange(min_val[i], max_val[i] + stp[i], stp[i])
                premises[i, :, 0] = h[:num_mfs]
                premises[i, :, 1] = stp[i]/2
        else:
            for i in range(input_size):
                premises[i, :, 0] = torch.mean(x_train[:, i])
                premises[i, :, 1] = torch.std(x_train[:, i])
                
        return premises
    
    def general_initialize_premises(self, x_train, mf_distribution):
        """
        Initializes the Gaussian MF premise parameters from training data, allowing a different number of MFs per input feature.
        
        Args:
            x_train (torch.Tensor): Training input data of shape ``(n_samples, input_size)``.
            mf_distribution (list[int]): Number of MFs for each input feature.
            
        Returns:
            list[torch.Tensor]: List of premise parameter tensors, one per input feature. The tensor at index ``i`` has shape ``(input_size, mf_distribution[i], len(self._params))``.
        """
        input_size = x_train.shape[1]
        premises = []
        
        for dim in range(input_size):
            num_mfs = mf_distribution[dim]
            dim_premises = torch.zeros(num_mfs, len(self._params), dtype=x_train.dtype)
            
            if num_mfs > 1:
                min_val = torch.min(x_train[:, dim])
                max_val = torch.max(x_train[:, dim])
                step = (max_val - min_val) / (num_mfs - 1)
                
                h = torch.arange(min_val, max_val + step, step)[:num_mfs]
                dim_premises[:,0] = h
                dim_premises[:,1] = step/2
            else:
                dim_premises[0,0] = torch.mean(x_train[:, dim])
                dim_premises[0,1] = torch.std(x_train[:, dim])
                
            premises.append(dim_premises)
            
        return premises
                
    
    def random_premises(self, input_size, num_mfs, dtype):
        """
        Generates randomly initialized premise parameters in the range ``[-1, 1]``, with :math:`\\sigma` constrained to positive values.
        
        Args:
            input_size (int): Number of input features.
            num_mfs (int): Number of MFs per input feature.
            dtype (torch.dtype): Data type of the premise parameters.
            
        Returns:
            torch.Tensor: Randomly initialized premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        """
        random_premises = 2 * torch.rand(input_size, num_mfs, len(self._params), dtype=dtype) - 1
        random_premises[:, :, 1] = torch.abs(random_premises[:, :, 1]) + 0.1
        return random_premises
    
    def random_single_feature_mfs(self, n_mfs, dtype):
        """
        Generates randomly initialized premise parameters for a single input feature in the range ``[-1, 1]``, with :math:`\\sigma` 
        constrained to positive values.
        
        Args:
            n_mfs (int): Number of MFs.
            dtype (torch.dtype): Data type of the premise parameters.
            
        Returns:
            torch.Tensor: Randomly initialized premise parameters of shape ``(n_mfs, len(self._params))``.
        """
        single_feature_mf = 2 * torch.rand(n_mfs, len(self._params), dtype=dtype) - 1
        single_feature_mf[:, 1] = torch.abs(single_feature_mf[:, 1]) + 0.1
        return single_feature_mf

    def _grow_new_premise_parameters(self, means, stds):
        """
        Generates new Gaussian MF premise parameters from provided means and standard 
        deviations, used during  structural adaptation in the SONFIS algorithm 
        (grow operation). Assumes the same number of MFs for every input feature.
        
        Args:
            means (torch.Tensor): Per-feature MF centers of shape ``(num_new_mfs, input_size)``, where ``num_new_mfs`` is the number of new MFs to add to each input feature.
            stds (torch.Tensor): Per-feature MF spreads of shape ``(num_new_mfs, input_size)``, matching the shape of ``means``.
            
        Returns:
            torch.Tensor: New premise parameters of shape ``(input_size, num_new_mfs, len(self._params))``.
        """
        return torch.cat((means.t().unsqueeze(2), stds.t().unsqueeze(2)), dim=2)
    
    def _split_premise_parameters(self, premises):
        """
        Splits each Gaussian MF in the given premises into two, used during structural adaptation in the SONFIS algorithm (split operation). Assumes the same number of MFs for every input feature.         
        
        Args:
            premises (torch.Tensor): Premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
            
        Returns:
            torch.Tensor: Resultant premise parameters of shape ``(input_size, 2 * num_mfs, len(self._params))``, where each original MF has been replaced by two new MFs.
        """
        split1 = torch.clone(premises)
        split1[:,:,0] += premises[:,:,1]/2
        split1[:,:,1] /= 2
        
        split2 = torch.clone(premises)
        split2[:,:,0] -= premises[:,:,1]/2
        split2[:,:,1] /= 2
        
        return torch.cat((split1, split2), dim=1)



class GeneralizedBell_MF(MembershipFunction):
    """
    Generalized bell-shaped membership function, defined as: 
    
    .. math::
    
        generalized\\_bell(x) = \\frac{1}{1 + \\left(\\frac{|x - c|}{a}\\right)^{2b}}
        
    where:
        - :math:`x` is the input variable.
        - :math:`a` is the width parameter.
        - :math:`b` is the slope parameter.
        - :math:`c` is the center parameter.
    
    """
    def __init__(self):
        super(GeneralizedBell_MF, self).__init__()
        self._params = ["a", "b", "c"] # ["width", "slope", "center"]

        self._min_val_plot = -2

        self._max_val_plot = 2
    
    def _simple_numpy_implementation(self, x, a, b, c):
        """
        NumPy implementation of the Generalized Bell MF, used for plotting.
        
        Args:
            x (numpy.ndarray): Input array.
            a (float): Width parameter.
            b (float): Slope parameter.
            c (float): Center parameter.
        
        Returns:
            numpy.ndarray: Membership degrees for each value in ``x``.
        
        """
        return 1/(1 + np.power(np.abs((x - c)/a), 2*b))
    
    def _simple_alpha_cut(self, alpha, a, b, c):
        """
        Computes the alpha-cut interval of the Generalized Bell MF.

        Args:
            alpha (float): Membership degree threshold in the range ``[0, 1]``.
            a (float): Width parameter.
            b (float): Slope parameter.
            c (float): Center parameter.
            
        Returns:
            numpy.ndarray: Array of 2 elements representing the interval over which the membership degree is greater than ``alpha``.
        """
        a = abs(a)
        r = np.power(1.0/alpha - 1.0, 1.0/(2.0*b))
        L = c - a*r
        U = c + a*r
        return np.array([L, U])
    
    def _get_center(self, a, b, c):
        """
        Returns the center of the Generalized Bell MF, used for plottin
        
        Args:
            a (float): Width parameter.
            b (float): Slope parameter.
            c (float): Center parameter.
            
        Returns:
            float: Center of the membership function.
        """
        return c
        
    def forward(self, x, premises):
        """
        Forward pass of the Generalized Bell membership function.
        
        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_size)``.
            premises (torch.Tensor): Premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
            
        Returns:
            torch.Tensor: Membership degrees of shape ``(batch_size, input_size, num_mfs)``.
        """
        return 1/(1 + torch.pow(torch.abs((x.unsqueeze(x.dim()) - premises[:, :, 2])/torch.where(premises[:, :, 0] == 0, torch.tensor(1e-6), premises[:, :, 0])), 2*premises[:, :, 1]))

    def initialize_premises(self, x_train, num_mfs):
        """
        Initializes the Generalized Bell MF premise parameters from training data, assuming the same number of MFs for every input feature.
        
        Args:
            x_train (torch.Tensor): Training input data of shape ``(n_samples, input_size)``.
            num_mfs (int): Number of MFs per input feature.
            
        Returns:
            torch.Tensor: Initialized premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        """
        input_size = x_train.shape[1]
        premises = torch.zeros(input_size, num_mfs, len(self._params), dtype=x_train.dtype)
        
        if num_mfs > 1:
            min_val = torch.min(x_train, dim=0).values
            max_val = torch.max(x_train, dim=0).values
            stp = (max_val - min_val) / (num_mfs - 1)
            for i in range(input_size):
                h = torch.arange(min_val[i], max_val[i] + stp[i], stp[i])
                premises[i, :, 2] = h[:num_mfs]
                premises[i, :, 0] = stp[i]/2
                premises[i, :, 1] = 4.
        else:
            for i in range(input_size):
                premises[i, :, 2] = torch.mean(x_train[:, i])
                premises[i, :, 0] = torch.std(x_train[:, i])
                premises[i, :, 1] = 4.
                
        return premises
    
    def general_initialize_premises(self, x_train, mf_distribution):
        """
        Initializes the Generalized Bell MF premise parameters from training data, allowing a different number of MFs per input feature.
        
        Args:
            x_train (torch.Tensor): Training input data of shape ``(n_samples, input_size)``.
            mf_distribution (list[int]): Number of MFs for each input feature.
            
        Returns:
            list[torch.Tensor]: List of premise parameter tensors, one per input feature. The tensor at index ``i`` has shape ``(input_size, mf_distribution[i], len(self._params))``.
        """
        input_size = x_train.shape[1]
        premises = []
        
        for dim in range(input_size):
            num_mfs = mf_distribution[dim]
            dim_premises = torch.zeros(num_mfs, len(self._params), dtype=x_train.dtype)
            
            if num_mfs > 1:
                min_val = torch.min(x_train[:, dim]).item()
                max_val = torch.max(x_train[:, dim]).item()
                step = (max_val - min_val) / (num_mfs - 1)
                
                h = torch.arange(min_val, max_val + step, step)[:num_mfs]
                dim_premises[:,2] = h
                dim_premises[:,0] = step/2
                dim_premises[:,1] = 4.
            else:
                dim_premises[0,2] = torch.mean(x_train[:, dim])
                dim_premises[0,0] = torch.std(x_train[:, dim])
                dim_premises[0,1] = 4.
                
            premises.append(dim_premises)
            
        return premises
    
    def random_premises(self, input_size, num_mfs, dtype):
        """
        Generates randomly initialized premise parameters in the range ``[-1, 1]``, with the width and slope parameters (:math:`a` and :math:`b`) constrained to positive values.
        
        Args:
            input_size (int): Number of input features.
            num_mfs (int): Number of MFs per input feature.
            dtype (torch.dtype): Data type of the premise parameters.
            
        Returns:
            torch.Tensor: Randomly initialized premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        """
        random_premises = 2 * torch.rand(input_size, num_mfs, len(self._params), dtype=dtype) - 1
        random_premises[:, :, :2] = torch.abs(random_premises[:, :, :2]) + 0.1
        random_premises[:, :, 1] += 2.
        return random_premises
    
    def random_single_feature_mfs(self, n_mfs, dtype):
        """
        Generates randomly initialized premise parameters for a single input feature in the range ``[-1, 1]``, with the width and slope parameters (:math:`a` and :math:`b`) constrained to positive values.
        
        Args:
            n_mfs (int): Number of MFs.
            dtype (torch.dtype): Data type of the premise parameters.
            
        Returns:
            torch.Tensor: Randomly initialized premise parameters of shape ``(n_mfs, len(self._params))``.
        """
        random_premise = 2 * torch.rand(n_mfs, len(self._params), dtype=dtype) - 1
        random_premise[:, :2] = torch.abs(random_premise[:, :2]) + 0.1
        random_premise[:, 1] += 2.
        return random_premise
    
    def _grow_new_premise_parameters(self, means, stds):
        """
        Generates new Generalized Bell MF premise parameters from provided means and standard deviations, 
        used during structural adaptation in the SONFIS algorithm (grow operation). Assumes the same number of MFs
        for every input feature. The slope parameter :math:`b` is initialized to ``4.0`` for all new MFs.
        
        Args:
            means (torch.Tensor): Per-feature MF centers of shape ``(num_new_mfs, input_size)``, where ``num_new_mfs`` is the number of new MFs to add to each input feature.
            stds (torch.Tensor): Per-feature MF spreads of shape ``(num_new_mfs, input_size)``, matching the shape of ``means``.
        
        Returns:
            torch.Tensor: New premise parameters of shape ``(input_size, num_new_mfs, len(self._params))``.
        """
        return torch.cat((stds.t().unsqueeze(2), (torch.ones_like(stds) * 4.).t().unsqueeze(2), means.t().unsqueeze(2)), dim=2)
    
    def _split_premise_parameters(self, premises):
        """
        Splits each Generalized Bell MF in the given premises into two, used during structural adaptation 
        in the SONFIS algorithm (split operation). Each MF is split by halving its width parameter :math:`a` and 
        shifting the center :math:`c` by the new half-width in opposite directions.
        Assumes the same number of MFs for every input feature.
        
        Args:
            premises (torch.Tensor): Premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
            
        Returns:
            torch.Tensor: Resultant premise parameters of shape ``(input_size, 2 * num_mfs, len(self._params))``, where each original MF has been replaced by two new MFs.
        """
        split1 = torch.clone(premises)
        split1[:,:,0] /= 2
        split1[:,:,2] += split1[:,:,0]
        
        split2 = torch.clone(premises)
        split2[:,:,0] /= 2
        split2[:,:,2] -= split2[:,:,0]
        
        return torch.cat((split1, split2), dim=1)



class HighSlopeBell_MF(MembershipFunction):
    """
    High-slope generalized bell-shaped membership function.

    A variant of the Generalized Bell MF where the slope parameter
    :math:`b` is fixed at ``8.0``, enforcing a strict and consistent
    shape across all MFs. It is defined as:
    
    .. math::
    
        generalized\\_bell(x) = \\frac{1}{1 + \\left(\\frac{|x - c|}{a}\\right)^{2\\cdot 8}}
        
    donde:
        - :math:`x` is the input variable.
        - :math:`a` is the width parameter.
        - :math:`c` is the center parameter.
    
    """
    def __init__(self):
        super(HighSlopeBell_MF, self).__init__()
        self._params = ["width", "center"] # ["width", "slope", "center"]

        self._min_val_plot = -2

        self._max_val_plot = 2
    
    def _simple_numpy_implementation(self, x, a, c):
        """
        NumPy implementation of the High-Slope Bell MF, used for plotting.
        
        Args:
            x (numpy.ndarray): Input array.
            a (float): Width parameter.
            c (float): Center parameter.
        
        Returns:
            numpy.ndarray: Membership degrees for each value in ``x``.
        
        """
        if a == 0:
            a = 1e-6
        return 1/(1 + np.power(np.abs((x - c)/a), 2*8.0))
    
    def _simple_alpha_cut(self, alpha, a, c):
        """
        Computes the alpha-cut interval of the High-Slope Bell MF.

        Args:
            alpha (float): Membership degree threshold in the range ``[0, 1]``.
            a (float): Width parameter.
            c (float): Center parameter.
            
        Returns:
            numpy.ndarray: Array of 2 elements representing the interval over which the membership degree is greater than ``alpha``.
        """
        a = abs(a)
        r = (1.0/alpha - 1.0)**(0.0625)
        L = c - a*r
        U = c + a*r
        return np.array([L, U])
    
    def _get_center(self, a, c):
        """
        Returns the center of the High-Slope Bell MF, used for plotting.
        
        Args:
            a (float): Width parameter.
            c (float): Center parameter.
            
        Returns:
            float: Center of the membership function.
        """
        return c
        
    def forward(self, x, premises):
        """
        Forward pass of the High-Slope Bell membership function.
        
        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_size)``.
            premises (torch.Tensor): Premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
            
        Returns:
            torch.Tensor: Membership degrees of shape ``(batch_size, input_size, num_mfs)``.
        """
        return 1/(1 + torch.pow(torch.abs((x.unsqueeze(x.dim()) - premises[:, :, 1])/torch.where(premises[:, :, 0] == 0, torch.tensor(1e-6), premises[:, :, 0])), 16.0))

    def initialize_premises(self, x_train, num_mfs):
        """
        Initializes the High-Slope Bell MF premise parameters from training data, assuming the same number of MFs for every input feature.
        
        Args:
            x_train (torch.Tensor): Training input data of shape ``(n_samples, input_size)``.
            num_mfs (int): Number of MFs per input feature.
            
        Returns:
            torch.Tensor: Initialized premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        """
        input_size = x_train.shape[1]
        premises = torch.zeros(input_size, num_mfs, len(self._params), dtype=x_train.dtype)
        
        if num_mfs > 1:
            min_val = torch.min(x_train, dim=0).values
            max_val = torch.max(x_train, dim=0).values
            stp = (max_val - min_val) / (num_mfs - 1)
            for i in range(input_size):
                h = torch.arange(min_val[i], max_val[i] + stp[i], stp[i])
                premises[i, :, 1] = h[:num_mfs]
                premises[i, :, 0] = stp[i]/2
        else:
            for i in range(input_size):
                premises[i, :, 1] = torch.mean(x_train[:, i])
                premises[i, :, 0] = torch.std(x_train[:, i])
                
        return premises
    
    def general_initialize_premises(self, x_train, mf_distribution):
        """
        Initializes the High-Slope Bell MF premise parameters from training data, allowing a different number of MFs per input feature.
        
        Args:
            x_train (torch.Tensor): Training input data of shape ``(n_samples, input_size)``.
            mf_distribution (list[int]): Number of MFs for each input feature.
            
        Returns:
            list[torch.Tensor]: List of premise parameter tensors, one per input feature. The tensor at index ``i`` has shape ``(input_size, mf_distribution[i], len(self._params))``.
        """
        input_size = x_train.shape[1]
        premises = []
        
        for dim in range(input_size):
            num_mfs = mf_distribution[dim]
            dim_premises = torch.zeros(num_mfs, len(self._params), dtype=x_train.dtype)
            
            if num_mfs > 1:
                min_val = torch.min(x_train[:, dim]).item()
                max_val = torch.max(x_train[:, dim]).item()
                step = (max_val - min_val) / (num_mfs - 1)
                
                h = torch.arange(min_val, max_val + step, step)[:num_mfs]
                dim_premises[:,1] = h
                dim_premises[:,0] = step/2
            else:
                dim_premises[0,1] = torch.mean(x_train[:, dim])
                dim_premises[0,0] = torch.std(x_train[:, dim])
                
            premises.append(dim_premises)
            
        return premises
    
    def random_premises(self, input_size, num_mfs, dtype):
        """
        Generates randomly initialized premise parameters in the range ``[-1, 1]``, with the width parameter (:math:`a`) constrained to positive values.
        
        Args:
            input_size (int): Number of input features.
            num_mfs (int): Number of MFs per input feature.
            dtype (torch.dtype): Data type of the premise parameters.
            
        Returns:
            torch.Tensor: Randomly initialized premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
        """
        random_premises = 2 * torch.rand(input_size, num_mfs, len(self._params), dtype=dtype) - 1
        random_premises[:, :, 0] = torch.abs(random_premises[:, :, 0]) + 0.1
        return random_premises
    
    def random_single_feature_mfs(self, n_mfs, dtype):
        """
        Generates randomly initialized premise parameters for a single input feature in the range ``[-1, 1]``, with the width parameter (:math:`a`) constrained to positive values.
        
        Args:
            n_mfs (int): Number of MFs.
            dtype (torch.dtype): Data type of the premise parameters.
            
        Returns:
            torch.Tensor: Randomly initialized premise parameters of shape ``(n_mfs, len(self._params))``.
        """
        random_premise = 2 * torch.rand(n_mfs, len(self._params), dtype=dtype) - 1
        random_premise[:, 0] = torch.abs(random_premise[:, 0]) + 0.1
        return random_premise
    
    def _grow_new_premise_parameters(self, means, stds):
        """
        Generates new High-Slope Bell MF premise parameters from provided means and standard deviations,
        used during structural adaptation in the SONFIS algorithm (grow operation). 
        Assumes the same number of MFs for every input feature.
        
        Args:
            means (torch.Tensor): Per-feature MF centers of shape ``(num_new_mfs, input_size)``, where ``num_new_mfs`` is the number of new MFs to add to each input feature.
            stds (torch.Tensor): Per-feature MF spreads of shape ``(num_new_mfs, input_size)``, matching the shape of ``means``.
            
        Returns:
            torch.Tensor: New premise parameters of shape ``(input_size, num_new_mfs, len(self._params))``.
        """
        return torch.cat((stds.t().unsqueeze(2), means.t().unsqueeze(2)), dim=2)
    
    def _split_premise_parameters(self, premises):
        """
        Splits each High-Slope Bell MF in the given premises into two, used during structural adaptation
        in the SONFIS algorithm (split operation). Each MF is split by halving its width parameter :math:`a` 
        and shifting the center :math:`c` by the new half-width in opposite directions.
        Assumes the same number of MFs for every input feature.
        
        Args:
            premises (torch.Tensor): Premise parameters of shape ``(input_size, num_mfs, len(self._params))``.
            
        Returns:
            torch.Tensor: Resultant premise parameters of shape ``(input_size, 2 * num_mfs, len(self._params))``, where each original MF has been replaced by two new MFs.
        """
        split1 = torch.clone(premises)
        split1[:,:,0] /= 2
        split1[:,:,1] += split1[:,:,0]
        
        split2 = torch.clone(premises)
        split2[:,:,0] /= 2
        split2[:,:,1] -= split2[:,:,0]
        
        return torch.cat((split1, split2), dim=1)