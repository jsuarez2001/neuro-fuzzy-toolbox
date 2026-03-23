import torch
import torch.nn as nn
import numpy as np

from abc import abstractmethod

class MembershipFunction(nn.Module):
    """
    Clase abstracta para funciones de membresía, solo define la estructura necesaria para diseñarlas en el toolbox. Sirve de guía para futuras implementaciones.
    """
    def __init__(self):
        super(MembershipFunction, self).__init__()
        self._params = None
        
        self._max_val_plot = None # For plotting
        self._min_val_plot = None # For plotting
        
    @abstractmethod
    def _simple_numpy_implementation(self, x, *args): # Simple numpy implementation needed for plotting
        """
        Implementación simple de la función de membresía en numpy. Este método se define para poder graficar las funciones de membresía.
        
        Args:
            x (numpy.ndarray): Tensor de entrada.
            *args: Parámetros de la función de membresía.
        """
        pass
    
    @abstractmethod
    def _simple_alpha_cut(self, alpha, *args):
        """
        Implementación simple de un alpha-cut.

        Args:
            alpha (float): Valor entre 0 y 1 que indica el grado de pertenencia mínimo que se desea usar para el corte
            *args: Parámetros de la función de membresía.

        Returns:
            np.array: Array de 2 elementos que representa el intervalo en el que el grado de pertenencia de la función es mayor a alpha.
        """
        pass
    
    @abstractmethod
    def _get_center(self, *args):
        """
        Método simple para retornar el centro de la función de membresía. Este método se define para poder graficar las funciones de membresía más facilmente.
        
        Args:
            *args: Parámetros de la función de membresía.
            
        Returns:
            float: Centro de la función de membresía.
        """
        pass
        
    @abstractmethod
    def forward(self, x, premises):
        """
        Paso hacia adelante de la función de membresía.
        
        Args:
            x (torch.tensor): Tensor de entrada. Debe tener la forma (batch_size, input_size).
            premises (torch.tensor): Parámetros de las premisas. Debe tener la forma (input_size, num_mfs, len(self._params)).
            
        Returns:
            torch.tensor: Salida de la función de membresía (valores de membresía). La forma de salida es (batch_size, input_size, num_mfs).
        """
        pass
    
    @abstractmethod
    def initialize_premises(self, x_train, num_mfs):
        """
        Inicializa los parámetros de la función de membresía basándose en los datos de entrada. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            x_train (torch.tensor): Conjunto de datos de entrenamiento de entrada. Debe tener la forma (n_samples, input_size).
            num_mfs (int): Número de funciones de membresía por característica de entrada.
            
        Returns:
            torch.tensor: Parámetros de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        pass
    
    @abstractmethod
    def general_initialize_premises(self, x_train, mf_distribution):
        """
        Inicializa los parámetros de la función de membresía basándose en los datos de entrada. Este método considera que cada característica de entrada puede tener un número distinto de funciones de membresía.
        
        Args:
            x_train (torch.tensor): Conjunto de datos de entrenamiento de entrada. Debe tener la forma (n_samples, input_size).
            mf_distribution (list): Número de funciones de membresía por característica de entrada en forma de lista.
            
        Returns:
            list: Lista de tensores con los parámetros de las funciones de membresía asociadas a cada feature de los datos de entrada. La forma del tensor i de la lista es (input_size, mf_distribution[i], len(self._params)).
        """
        pass
    
    @abstractmethod
    def random_premises(self, input_size, num_mfs, dtype):
        """
        Genera parámetros aleatorios para las premisas en el rango [-1, 1], restringiendo ciertos parámetros a valores positivos (dependiendo de la función de membresía en cuestión).
        
        Args:
            input_size (int): Número de características de entrada.
            num_mfs (int): Número de funciones de membresía por característica de entrada.
            dtype (torch.dtype): Tipo de dato de las premisas.
            
        Returns:
            torch.tensor: Parámetros aleatorios de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        pass
    
    @abstractmethod
    def random_single_feature_mfs(self):
        """
        Genera parámetros aleatorios para una sola función de membresía en el rango [-1, 1], restringiendo ciertos parámetros a valores positivos (dependiendo de la función de membresía en cuestión).
        
        Args:
            n_mfs (int): Número de funciones de membresía.
            dtype (torch.dtype): Tipo de dato de las premisas.
            
        Returns:
            torch.tensor: Parámetros aleatorios de las premisas. La forma de salida es (n_mfs, len(self._params)).
        """
        pass
    
    @abstractmethod
    def _grow_new_premise_parameters(self, means, stds):
        """
        Método utilizado en el algoritmo para la modificación de la estructura del modelo SONFIS. Genera nuevas premisas dado un conjunto de medias y desviaciones estándar en la forma de un tensor. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            means (torch.tensor): Medias por característica de entrada y función de membresía. La dimensión de la entrada es (num_new_mfs, input_size), donde num_new_mfs es el número de nuevas funciones de membresía que se agregarán a cada feature de los datos de entrada.
            stds (torch.tensor): Desviaciones estándar por característica de entrada y función de membresía. La dimensión de la entrada es (num_new_mfs, input_size), de la misma forma que el tensor means.
            
        Returns:
            torch.tensor: Nuevos parámetros de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        pass
    
    @abstractmethod
    def _split_premise_parameters(self, premises):
        """
        Método utilizado en el algoritmo para la modificación de la estructura del modelo SONFIS. Divide las premisas ingresadas en dos nuevas. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            premises (torch.tensor): Parámetros de las premisas. La forma de entrada es (input_size, num_mfs, len(self._params)).
            
        Returns:
            torch.tensor: Nuevos parámetros de las premisas. La forma de salida es (input_size, 2*num_mfs, len(self._params)).
        """
        pass


class Gaussian_MF(MembershipFunction):
    """  
    Función de membresía Gaussiana. Está definida como:  

    .. math::

        gaussian(x) = e^{-\\frac{(x - \\mu)^2}{2\\sigma^2}}

    donde:
        - :math:`x` es la variable de entrada.
        - :math:`\\mu` es el centro de la función.
        - :math:`\\sigma` es la desviación estándar, que controla la anchura de la curva.
    
    """
    def __init__(self):
        super(Gaussian_MF, self).__init__()
        self._params = ["mu", "sigma"]
        
        self._min_val_plot = -2
        self._max_val_plot = 2
        
    def _simple_numpy_implementation(self, x, mu, sigma):
        """
        Implementación simple de la función de membresía Gaussiana en numpy. Este método se define para poder graficar las funciones de membresía.
        
        Args:
            x (numpy.ndarray): Tensor de entrada.
            mu (float): Parámetro mu.
            sigma (float): Parámetro sigma.
            
        Returns:
            numpy.ndarray: Salida de la función de membresía Gaussiana.
        """
        return np.exp(-0.5 * np.power((x - mu)/sigma, 2))
    
    def _simple_alpha_cut(self, alpha, mu, sigma):
        """
        Implementación simple de un alpha-cut.

        Args:
            alpha (float): Valor entre 0 y 1 que indica el grado de pertenencia mínimo que se desea usar para el corte
            mu (float): Parámetro mu.
            sigma (float): Parámetro sigma.

        Returns:
            np.array: Array de 2 elementos que representa el intervalo en el que el grado de pertenencia de la función es mayor a alpha.
        """
        sigma = abs(sigma)
        r = np.sqrt(-2*np.log(alpha))
        L = mu - sigma*r
        U = mu + sigma*r
        return np.array([L, U])
    
    def _get_center(self, mu, sigma):
        """
        Método simple para retornar el centro de la función de membresía. Este método se define para poder graficar las funciones de membresía más facilmente.
        
        Args:
            mu (float): Parámetro mu.
            sigma (float): Parámetro sigma.
            
        Returns:
            float: Centro de la función de membresía.
        """
        return mu

    def forward(self, x, premises):
        """
        Paso hacia adelante de la función de membresía Gaussiana.
        
        Args:
            x (torch.tensor): Tensor de entrada. Debe tener la forma (batch_size, input_size).
            premises (torch.tensor): Parámetros de las premisas. Debe tener la forma (input_size, num_mfs, len(self._params)).
        
        Returns:
            torch.tensor: Salida de la función de membresía Gaussiana (valores de membresía). La forma de salida es (batch_size, input_size, num_mfs).
        """
        return torch.exp(-0.5 * torch.pow((x.unsqueeze(x.dim()) - premises[:, :, 0])/torch.where(premises[:, :, 1] == 0, torch.tensor(1e-6), premises[:, :, 1]), 2))

    def initialize_premises(self, x_train, num_mfs):
        """
        Inicializa los parámetros de la función de membresía gaussiana basándose en los datos de entrada. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            x_train (torch.tensor): Conjunto de datos de entrenamiento de entrada. Debe tener la forma (n_samples, input_size).
            num_mfs (int): Número de funciones de membresía por característica de entrada.
            
        Returns:
            torch.tensor: Parámetros de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
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
        Inicializa los parámetros de la función de membresía gaussiana basándose en los datos de entrada. Este método considera que cada característica de entrada puede tener un número distinto de funciones de membresía.
        
        Args:
            x_train (torch.tensor): Conjunto de datos de entrenamiento de entrada. Debe tener la forma (n_samples, input_size).
            mf_distribution (list): Número de funciones de membresía por característica de entrada en forma de lista.
            
        Returns:
            list: Lista de tensores con los parámetros de las funciones de membresía asociadas a cada feature de los datos de entrada. La forma del tensor i de la lista es (input_size, mf_distribution[i], len(self._params)).
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
        Genera parámetros aleatorios para las premisas en el rango [-1, 1], restringiendo el parámetro :math:`\\sigma` a valores positivos.
        
        Args:
            input_size (int): Número de características de entrada.
            num_mfs (int): Número de funciones de membresía por característica de entrada.
            dtype (torch.dtype): Tipo de dato de las premisas.
            
        Returns:
            torch.tensor: Parámetros aleatorios de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        random_premises = 2 * torch.rand(input_size, num_mfs, len(self._params), dtype=dtype) - 1
        random_premises[:, :, 1] = torch.abs(random_premises[:, :, 1]) + 0.1
        return random_premises
    
    def random_single_feature_mfs(self, n_mfs, dtype):
        """
        Genera parámetros aleatorios para una sola función de membresía Gaussiana en el rango [-1, 1], restringiendo el parámetro :math:`\\sigma` a valores positivos.
        
        Args:
            n_mfs (int): Número de funciones de membresía.
            dtype (torch.dtype): Tipo de dato de las premisas.
            
        Returns:
            torch.tensor: Parámetros aleatorios de las premisas. La forma de salida es (n_mfs, len(self._params)).
        """
        single_feature_mf = 2 * torch.rand(n_mfs, len(self._params), dtype=dtype) - 1
        single_feature_mf[:, 1] = torch.abs(single_feature_mf[:, 1]) + 0.1
        return single_feature_mf

    def _grow_new_premise_parameters(self, means, stds):
        """
        Método utilizado en el algoritmo para la modificación de la estructura del modelo SONFIS. Genera nuevas premisas dado un conjunto de medias y desviaciones estándar en la forma de un tensor. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            means (torch.tensor): Medias por característica de entrada y función de membresía. La dimensión de la entrada es (num_new_mfs, input_size), donde num_new_mfs es el número de nuevas funciones de membresía que se agregarán a cada feature de los datos de entrada.
            stds (torch.tensor): Desviaciones estándar por característica de entrada y función de membresía. La dimensión de la entrada es (num_new_mfs, input_size), de la misma forma que el tensor means.
            
        Returns:
            torch.tensor: Nuevos parámetros de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        return torch.cat((means.t().unsqueeze(2), stds.t().unsqueeze(2)), dim=2)
    
    def _split_premise_parameters(self, premises):
        """
        Método utilizado en el algoritmo para la modificación de la estructura del modelo SONFIS. Divide las premisas ingresadas en dos nuevas. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            premises (torch.tensor): Parámetros de las premisas. La forma de entrada es (input_size, num_mfs, len(self._params)).
            
        Returns:
            torch.tensor: Nuevos parámetros de las premisas. La forma de salida es (input_size, 2*num_mfs, len(self._params)).
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
    Función de membresía de tipo generalized bell-shaped. Está definida como:  
    
    .. math::
    
        generalized\\_bell(x) = \\frac{1}{1 + \\left(\\frac{|x - c|}{a}\\right)^{2b}}
        
    donde:
        - :math:`x` es la variable de entrada.
        - :math:`a` es el parámetro de ancho.
        - :math:`b` es el parámetro de pendiente.
        - :math:`c` es el parámetro de centro.
    
    """
    def __init__(self):
        super(GeneralizedBell_MF, self).__init__()
        self._params = ["a", "b", "c"] # ["width", "slope", "center"]

        self._min_val_plot = -2

        self._max_val_plot = 2
    
    def _simple_numpy_implementation(self, x, a, b, c):
        """
        Implementación simple de la función de membresía Generalized Bell en numpy. Este método se define para poder graficar las funciones de membresía.
        
        Args:
            x (numpy.ndarray): Tensor de entrada.
            a (float): Parámetro a (ancho).
            b (float): Parámetro b (pendiente).
            c (float): Parámetro c (centro).
        
        Returns:
            numpy.ndarray: Salida de la función de membresía Gaussiana.
        
        """
        return 1/(1 + np.power(np.abs((x - c)/a), 2*b))
    
    def _simple_alpha_cut(self, alpha, a, b, c):
        """
        Implementación simple de un alpha-cut.

        Args:
            alpha (float): Valor entre 0 y 1 que indica el grado de pertenencia mínimo que se desea usar para el corte
            a (float): Parámetro a (ancho).
            b (float): Parámetro b (pendiente).
            c (float): Parámetro c (centro).
            
        Returns:
            np.array: Array de 2 elementos que representa el intervalo en el que el grado de pertenencia de la función es mayor a alpha.
        """
        a = abs(a)
        r = np.power(1.0/alpha - 1.0, 1.0/(2.0*b))
        L = c - a*r
        U = c + a*r
        return np.array([L, U])
    
    def _get_center(self, a, b, c):
        """
        Método simple para retornar el centro de la función de membresía. Este método se define para poder graficar las funciones de membresía más facilmente.
        
        Args:
            a (float): Parámetro a (ancho).
            b (float): Parámetro b (pendiente).
            c (float): Parámetro c (centro).
            
        Returns:
            float: Centro de la función de membresía.
        """
        return c
        
    def forward(self, x, premises):
        """
        Paso hacia adelante de la función de membresía Generalized Bell.
        
        Args:
            x (torch.tensor): Tensor de entrada. Debe tener la forma (batch_size, input_size).
            premises (torch.tensor): Parámetros de las premisas. Debe tener la forma (input_size, num_mfs, len(self._params)).
            
        Returns:
            torch.tensor: Salida de la función de membresía Generalized Bell (valores de membresía). La forma de salida es (batch_size, input_size, num_mfs).
        """
        return 1/(1 + torch.pow(torch.abs((x.unsqueeze(x.dim()) - premises[:, :, 2])/torch.where(premises[:, :, 0] == 0, torch.tensor(1e-6), premises[:, :, 0])), 2*premises[:, :, 1]))

    def initialize_premises(self, x_train, num_mfs):
        """
        Inicializa los parámetros de la función de membresía Generalized Bell basándose en los datos de entrada. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            x_train (torch.tensor): Conjunto de datos de entrenamiento de entrada. Debe tener la forma (n_samples, input_size).
            num_mfs (int): Número de funciones de membresía por característica de entrada.
            
        Returns:
            torch.tensor: Parámetros de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
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
        Inicializa los parámetros de la función de membresía Generalized Bell basándose en los datos de entrada. Este método considera que cada característica de entrada puede tener un número distinto de funciones de membresía.
        
        Args:
            x_train (torch.tensor): Conjunto de datos de entrenamiento de entrada. Debe tener la forma (n_samples, input_size).
            mf_distribution (list): Número de funciones de membresía por característica de entrada en forma de lista.
            
        Returns:
            list: Lista de tensores con los parámetros de las funciones de membresía asociadas a cada feature de los datos de entrada. La forma del tensor i de la lista es (input_size, mf_distribution[i], len(self._params)).
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
        Genera parámetros aleatorios para las premisas en el rango [-1, 1], restringiendo a los parámetros de ancho y pendiente (**a** y **b**) a valores positivos. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            input_size (int): Número de características de entrada.
            num_mfs (int): Número de funciones de membresía por característica de entrada.
            dtype (torch.dtype): Tipo de dato de las premisas.
            
        Returns:
            torch.tensor: Parámetros aleatorios de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        random_premises = 2 * torch.rand(input_size, num_mfs, len(self._params), dtype=dtype) - 1
        random_premises[:, :, :2] = torch.abs(random_premises[:, :, :2]) + 0.1
        random_premises[:, :, 1] += 2.
        return random_premises
    
    def random_single_feature_mfs(self, n_mfs, dtype):
        """
        Genera parámetros aleatorios para una sola función de membresía Generalized Bell en el rango [-1, 1], restringiendo a los parámetros de ancho y pendiente (**a** y **b**) a valores positivos.
        
        Args:
            n_mfs (int): Número de funciones de membresía.
            dtype (torch.dtype): Tipo de dato de las premisas.
            
        Returns:
            torch.tensor: Parámetros aleatorios de las premisas. La forma de salida es (n_mfs, len(self._params)).
        """
        random_premise = 2 * torch.rand(n_mfs, len(self._params), dtype=dtype) - 1
        random_premise[:, :2] = torch.abs(random_premise[:, :2]) + 0.1
        random_premise[:, 1] += 2.
        return random_premise
    
    def _grow_new_premise_parameters(self, means, stds):
        """
        Método utilizado en el algoritmo para la modificación de la estructura del modelo SONFIS. Genera nuevas premisas dado un conjunto de medias y desviaciones estándar en la forma de un tensor. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            means (torch.tensor): Medias por característica de entrada y función de membresía. La dimensión de la entrada es (num_new_mfs, input_size), donde num_new_mfs es el número de nuevas funciones de membresía que se agregarán a cada feature de los datos de entrada.
            stds (torch.tensor): Desviaciones estándar por característica de entrada y función de membresía. La dimensión de la entrada es (num_new_mfs, input_size), de la misma forma que el tensor means.
            
        Returns:
            torch.tensor: Nuevos parámetros de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        return torch.cat((stds.t().unsqueeze(2), (torch.ones_like(stds) * 4.).t().unsqueeze(2), means.t().unsqueeze(2)), dim=2)
    
    def _split_premise_parameters(self, premises):
        """
        Método utilizado en el algoritmo para la modificación de la estructura del modelo SONFIS. Divide las premisas ingresadas en dos nuevas. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            premises (torch.tensor): Parámetros de las premisas. La forma de entrada es (input_size, num_mfs, len(self._params)).
            
        Returns:
            torch.tensor: Nuevos parámetros de las premisas. La forma de salida es (input_size, 2*num_mfs, len(self._params)).
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
    Función de membresía de tipo generalized bell-shaped, pero el parámetro "b" relacionado con la "pendiente" de la campana está definido a 8.0 para establecer una forma determinada y estricta a las funciones de membresía. Está definida como:  
    
    .. math::
    
        generalized\\_bell(x) = \\frac{1}{1 + \\left(\\frac{|x - c|}{a}\\right)^{2\\cdot 8}}
        
    donde:
        - :math:`x` es la variable de entrada.
        - :math:`a` es el parámetro de ancho.
        - :math:`c` es el parámetro de centro.
    
    """
    def __init__(self):
        super(HighSlopeBell_MF, self).__init__()
        self._params = ["width", "center"] # ["width", "slope", "center"]

        self._min_val_plot = -2

        self._max_val_plot = 2
    
    def _simple_numpy_implementation(self, x, a, c):
        """
        Implementación simple de la función de membresía High Slope Bell en numpy. Este método se define para poder graficar las funciones de membresía.
        
        Args:
            x (numpy.ndarray): Tensor de entrada.
            a (float): Parámetro a (ancho).
            c (float): Parámetro c (centro).
        
        Returns:
            numpy.ndarray: Salida de la función de membresía Gaussiana.
        
        """
        if a == 0:
            a = 1e-6
        return 1/(1 + np.power(np.abs((x - c)/a), 2*8.0))
    
    def _simple_alpha_cut(self, alpha, a, c):
        """
        Implementación simple de un alpha-cut.

        Args:
            alpha (float): Valor entre 0 y 1 que indica el grado de pertenencia mínimo que se desea usar para el corte
            a (float): Parámetro a (ancho).
            c (float): Parámetro c (centro).
            
        Returns:
            np.array: Array de 2 elementos que representa el intervalo en el que el grado de pertenencia de la función es mayor a alpha.
        """
        a = abs(a)
        r = (1.0/alpha - 1.0)**(0.0625)
        L = c - a*r
        U = c + a*r
        return np.array([L, U])
    
    def _get_center(self, a, c):
        """
        Método simple para retornar el centro de la función de membresía. Este método se define para poder graficar las funciones de membresía más facilmente.
        
        Args:
            a (float): Parámetro a (ancho).
            c (float): Parámetro c (centro).
            
        Returns:
            float: Centro de la función de membresía.
        """
        return c
        
    def forward(self, x, premises):
        """
        Paso hacia adelante de la función de membresía High Slope Bell.
        
        Args:
            x (torch.tensor): Tensor de entrada. Debe tener la forma (batch_size, input_size).
            premises (torch.tensor): Parámetros de las premisas. Debe tener la forma (input_size, num_mfs, len(self._params)).
            
        Returns:
            torch.tensor: Salida de la función de membresía High Slope Bell (valores de membresía). La forma de salida es (batch_size, input_size, num_mfs).
        """
        return 1/(1 + torch.pow(torch.abs((x.unsqueeze(x.dim()) - premises[:, :, 1])/torch.where(premises[:, :, 0] == 0, torch.tensor(1e-6), premises[:, :, 0])), 16.0))

    def initialize_premises(self, x_train, num_mfs):
        """
        Inicializa los parámetros de la función de membresía High Slope Bell basándose en los datos de entrada. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            x_train (torch.tensor): Conjunto de datos de entrenamiento de entrada. Debe tener la forma (n_samples, input_size).
            num_mfs (int): Número de funciones de membresía por característica de entrada.
            
        Returns:
            torch.tensor: Parámetros de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
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
        Inicializa los parámetros de la función de membresía High Slope Bell basándose en los datos de entrada. Este método considera que cada característica de entrada puede tener un número distinto de funciones de membresía.
        
        Args:
            x_train (torch.tensor): Conjunto de datos de entrenamiento de entrada. Debe tener la forma (n_samples, input_size).
            mf_distribution (list): Número de funciones de membresía por característica de entrada en forma de lista.
            
        Returns:
            list: Lista de tensores con los parámetros de las funciones de membresía asociadas a cada feature de los datos de entrada. La forma del tensor i de la lista es (input_size, mf_distribution[i], len(self._params)).
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
        Genera parámetros aleatorios para las premisas en el rango [-1, 1], restringiendo el parámetro del ancho (**width**) a valores positivos. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            input_size (int): Número de características de entrada.
            num_mfs (int): Número de funciones de membresía por característica de entrada.
            dtype (torch.dtype): Tipo de dato de las premisas.
            
        Returns:
            torch.tensor: Parámetros aleatorios de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        random_premises = 2 * torch.rand(input_size, num_mfs, len(self._params), dtype=dtype) - 1
        random_premises[:, :, 0] = torch.abs(random_premises[:, :, 0]) + 0.1
        return random_premises
    
    def random_single_feature_mfs(self, n_mfs, dtype):
        """
        Genera parámetros aleatorios para una sola función de membresía High Slope Bell en el rango [-1, 1], restringiendo el parámetro del ancho (**width**) a valores positivos.
        
        Args:
            n_mfs (int): Número de funciones de membresía.
            dtype (torch.dtype): Tipo de dato de las premisas.
            
        Returns:
            torch.tensor: Parámetros aleatorios de las premisas. La forma de salida es (n_mfs, len(self._params)).
        """
        random_premise = 2 * torch.rand(n_mfs, len(self._params), dtype=dtype) - 1
        random_premise[:, 0] = torch.abs(random_premise[:, 0]) + 0.1
        return random_premise
    
    def _grow_new_premise_parameters(self, means, stds):
        """
        Método utilizado en el algoritmo para la modificación de la estructura del modelo SONFIS. Genera nuevas premisas dado un conjunto de medias y desviaciones estándar en la forma de un tensor. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            means (torch.tensor): Medias por característica de entrada y función de membresía. La dimensión de la entrada es (num_new_mfs, input_size), donde num_new_mfs es el número de nuevas funciones de membresía que se agregarán a cada feature de los datos de entrada.
            stds (torch.tensor): Desviaciones estándar por característica de entrada y función de membresía. La dimensión de la entrada es (num_new_mfs, input_size), de la misma forma que el tensor means.
            
        Returns:
            torch.tensor: Nuevos parámetros de las premisas. La forma de salida es (input_size, num_mfs, len(self._params)).
        """
        return torch.cat((stds.t().unsqueeze(2), means.t().unsqueeze(2)), dim=2)
    
    def _split_premise_parameters(self, premises):
        """
        Método utilizado en el algoritmo para la modificación de la estructura del modelo SONFIS. Divide las premisas ingresadas en dos nuevas. Este método considera que todas las características de entrada tienen el mismo número de funciones de membresía.
        
        Args:
            premises (torch.tensor): Parámetros de las premisas. La forma de entrada es (input_size, num_mfs, len(self._params)).
            
        Returns:
            torch.tensor: Nuevos parámetros de las premisas. La forma de salida es (input_size, 2*num_mfs, len(self._params)).
        """
        split1 = torch.clone(premises)
        split1[:,:,0] /= 2
        split1[:,:,1] += split1[:,:,0]
        
        split2 = torch.clone(premises)
        split2[:,:,0] /= 2
        split2[:,:,1] -= split2[:,:,0]
        
        return torch.cat((split1, split2), dim=1)