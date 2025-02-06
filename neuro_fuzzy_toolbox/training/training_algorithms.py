import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split

from neuro_fuzzy_toolbox.training import (
    classical_consequents_estimation_with_OLS,
    optimizer_training_epoch
)

class base_model_trainer():
    """
    Clase base para representar un algoritmo de entrenamiento para un modelo de aprendizaje automático.
    
    Warning:
        Esta clase no debe ser instanciada directamente. Solo sirve para ser heredada por otras clases. 
        
    Note:
        - Las clases que hereden de esta clase deben implementar las funciones _update_parameters(model, loader) e init_optimizer(self, model, optimizer, optim_params).
        - La función _update_parameters(model, loader) ejecuta la actualización de parámetros durante una sola época el entrenamiento del modelo.
        - La función init_optimizer(self, model, optimizer, optim_params) instancia el optimizador a utilizar durante el entrenamiento del modelo.
        
    """
    
    def __init__(self, epochs, loss_function, validation=0, early_stopping=None, optimizer=torch.optim.Adam, optimizer_params={}):
        """
        Inicializa una nueva instancia de la clase base_model_trainer.
        
        Args:
            epochs (int): Número de épocas de entrenamiento.
            loss_function (torch.nn.functional): Función de pérdida a utilizar en el entrenamiento.
            validation (float): Proporción de los datos de entrenamiento a utilizar como datos de validación (Default: 0).
            early_stopping (EarlyStopping): Mecanismo de Early Stopping a utilizar en el entrenamiento (Default: None).
            optimizer (torch.optim): Optimizador a utilizar en el entrenamiento (Default: torch.optim.Adam).
            optimizer_params (dict): Parámetros a utilizar en el optimizador (Default: {}).
        
        """
        # Training parameters
        self.epochs = epochs
        self.loss_function = loss_function
        
        # Optimizer
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self._optimizer_instance = None
        
        # Early stopping
        self.validation = validation
        self.early_stopping = early_stopping
        
        # History
        self.history = {"loss": []}
        self.val_history = {"loss": []}

        
    def __call__(self, model, loader, verbose=True):
        """
        Entrena un modelo utilizando el algoritmo de entrenamiento.
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
            loader (DataLoader): DataLoader con los datos de entrenamiento.
            verbose (bool): Indica si se deben imprimir mensajes de aviso (Default: True).
        
        """
        train_loader, val_loader = self._train_val_split(loader)
        
        self._optimizer_instance = self._init_optimizer(model, self.optimizer, self.optimizer_params)
        
        self._register_loss(model, train_loader, val_loader)

        ep = 0
        while ep < self.epochs:
            
            self._update_parameters(model, train_loader)
            
            self._register_loss(model, train_loader, val_loader)
            
            loss, val_loss = self.history["loss"][-1], self.val_history["loss"][-1]
            
            if self.validation > 0 and self._check_early_stop(model, val_loss, verbose):
                break
                
            if verbose:
                epoch_width = len(str(self.epochs))
                if self.validation > 0:
                    print(f'Epoch: {ep+1:{epoch_width}}/{self.epochs} - loss: {loss:.6f} - validation loss: {val_loss:.6f}')
                else:
                    print(f'Epoch: {ep+1:{epoch_width}}/{self.epochs} - loss: {loss:.6f}')
                    
            ep += 1
        
        if verbose:
            print('Training finished')
            
            
    def _loss_function(self, pred, y):
        """
        Ejecuta la función de pérdida.
        
        Args:
            pred (torch.Tensor): Tensor con las predicciones del modelo.
            y (torch.Tensor): Tensor con las etiquetas reales.
            
        Returns:
            torch.Tensor: Tensor con el valor de la pérdida.
        """
        
        '''preliminary fix for the dtype issue'''
        if y.dtype != pred.dtype:
            y = y.to(pred.dtype)
        '''preliminary fix for the dtype issue'''
        
        '''torch cross_entropy function only accepts torch.long (torch.int64) dtype for target indices'''
        if self.loss_function == torch.nn.functional.cross_entropy:
            y = y.type(torch.int64)
        '''torch cross_entropy function only accepts torch.long (torch.int64) dtype for target indices'''
            
        return self.loss_function(pred, y)
    
    def _train_val_split(self, train_loader):
        """
        Divide los datos de entrenamiento en datos de entrenamiento y validación.
        
        Args:
            train_loader (DataLoader): DataLoader con los datos de entrenamiento.
            
        Returns:
            DataLoader: DataLoader con los datos de entrenamiento.
            DataLoader: DataLoader con los datos de validación.
        """
        val_loader = None
        
        if self.validation != 0:
            x_train, y_train = train_loader.dataset.tensors
            x_train, x_val, y_train, y_val = train_test_split(x_train.numpy(), y_train.numpy(), test_size=self.validation, shuffle=True)
            x_train, x_val, y_train, y_val = torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(y_train), torch.from_numpy(y_val)
            
            train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), batch_size=train_loader.batch_size, shuffle=True)
            val_loader = data.DataLoader(data.TensorDataset(x_val, y_val), batch_size=train_loader.batch_size, shuffle=False)

        return train_loader, val_loader
    
    
    def _check_early_stop(self, model, loss, verbose):
        """
        Verifica si se debe detener el entrenamiento del modelo (si hay mecanismo EarlyStopping).
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
            loss (float): Pérdida actual del modelo.
            verbose (bool): Indica si se deben imprimir mensajes de aviso.
            
        Returns:
            bool: Indica si se debe detener el entrenamiento.
        """
        if self.early_stopping is not None:
            self.early_stopping(model, loss, verbose)
            if self.early_stopping.stop:
                return True
        return False
    
    
    def _loss(self, model, train_loader, val_loader):
        """
        Calcula el valor de la pérdida del modelo.
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
            train_loader (DataLoader): DataLoader con los datos de entrenamiento.
            val_loader (DataLoader): DataLoader con los datos de validación.
        
        Returns:
            torch.Tensor: Tensor con el valor de la pérdida.
            torch.Tensor: Tensor con el valor de la pérdida de validación.
        """
        val_loss = None
        x = train_loader.dataset.tensors[0]
        y = train_loader.dataset.tensors[1]
        with torch.no_grad():
            pred = model(x)
            loss = self._loss_function(pred, y)
        if self.validation > 0:
            x = val_loader.dataset.tensors[0]
            y = val_loader.dataset.tensors[1]
            with torch.no_grad():
                pred = model(x)
                val_loss = self._loss_function(pred, y)
        return loss, val_loss
    
    
    def _register_loss(self, model, train_loader, val_loader):
        """
        Registra el valor de la pérdida en el historial. Si hay validación, también registra la pérdida de validación.
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
            train_loader (DataLoader): DataLoader con los datos de entrenamiento.
            val_loader (DataLoader): DataLoader con los datos de valid
        
        """
        loss, val_loss = self._loss(model, train_loader, val_loader)
        self.history["loss"].append(loss.item())
        if self.validation > 0:
            self.val_history["loss"].append(val_loss.item())



class Hybrid_learning_algorithm(base_model_trainer):
    """
    Clase para representar un algoritmo de entrenamiento híbrido para un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS).
    
    Pseudocódigo:
        .. code-block:: text

            testes test:
                Para cada test en test:
                    Calcular test de membresía usando la función de test
                    Guardar en el tensor de test
            Retornar el test con los valores de test
    """
    def __init__(self, epochs, loss_function, validation=0, early_stopping=None, optimizer=torch.optim.Adam, optimizer_params={}):
        """
        Inicializa una nueva instancia de la clase Hybrid_learning_algorithm.
        
        Args:
            epochs (int): Número de épocas de entrenamiento.
            loss_function (torch.nn.functional): Función de pérdida a utilizar en el entrenamiento.
            validation (float): Proporción de los datos de entrenamiento a utilizar como datos de validación (Default: 0).
            early_stopping (EarlyStopping): Mecanismo de Early Stopping a utilizar en el entrenamiento (Default: None).
            optimizer (torch.optim): Optimizador a utilizar en el entrenamiento (Default: torch.optim.Adam).
            optimizer_params (dict): Parámetros a utilizar en el optimizador (Default: {}).
        
        """
        super().__init__(epochs, loss_function, validation, early_stopping, optimizer, optimizer_params)
        
        
    def _init_optimizer(self, model, optimizer, optim_params):
        """
        Inicializa el optimizador a utilizar en el entrenamiento, utilizando los parámetros de los antececentes del modelo ANFIS y los parámetros del optimizador especificados.
        
        Args:
            model (ANFIS | h_ANFIS): Modelo ANFIS a entrenar.
            optimizer (torch.optim): Optimizador a utilizar en el entrenamiento.
            optim_params (dict): Parámetros a utilizar en el optimizador.
            
        Returns:
            torch.optim.Optimizer: Instancia del optimizador.
        """
        return optimizer(model.get_premises_as_parameters_list(), **optim_params)
    
    
    def _premises_update(self, ANFISmodel, loader):
        """
        Actualiza los parámetros de los antecedentes del modelo ANFIS.
        
        Args:
            ANFISmodel (ANFIS | h_ANFIS): Modelo ANFIS a entrenar.
            loader (DataLoader): DataLoader con los datos de entrenamiento.
        """
        optimizer_training_epoch(ANFISmodel, loader, self._optimizer_instance, self.loss_function)
        
    
    def _consequents_update(self, ANFISmodel, loader):
        """
        Actualiza los parámetros consecuentes del modelo ANFIS.
        
        Args:
            ANFISmodel (ANFIS | h_ANFIS): Modelo ANFIS a entrenar.
            loader (DataLoader): DataLoader con los datos de entrenamiento.
        """
        ANFISmodel.set_consequents(classical_consequents_estimation_with_OLS(ANFISmodel, loader))
    

    def _update_parameters(self, ANFISmodel, loader):
        """
        Actualiza los parámetros del modelo ANFIS (época única).
        
        Args:
            ANFISmodel (ANFIS | h_ANFIS): Modelo ANFIS a entrenar.
            loader (DataLoader): DataLoader con los datos de entrenamiento.
        """
        self._consequents_update(ANFISmodel, loader)
        self._premises_update(ANFISmodel, loader)



class Optimizer_training(base_model_trainer):
    """
    Clase para representar un algoritmo de entrenamiento por optimización para un modelo de aprendizaje automático.
    
    Pseudocódigo:
        .. code-block:: text

            testes test:
                Para cada test en test:
                    Calcular test de membresía usando la función de test
                    Guardar en el tensor de test
            Retornar el test con los valores de test
    """
    
    def __init__(self, epochs, loss_function, validation=0, early_stopping=None, optimizer=torch.optim.Adam, optimizer_params={}):
        """
        Inicializa una nueva instancia de la clase Optimizer_training.
        
        Args:
            epochs (int): Número de épocas de entrenamiento.
            loss_function (torch.nn.functional): Función de pérdida a utilizar en el entrenamiento.
            validation (float): Proporción de los datos de entrenamiento a utilizar como datos de validación (Default: 0).
            early_stopping (EarlyStopping): Mecanismo de Early Stopping a utilizar en el entrenamiento (Default: None).
            optimizer (torch.optim): Optimizador a utilizar en el entrenamiento (Default: torch.optim.Adam).
            optimizer_params (dict): Parámetros a utilizar en el optimizador (Default: {}).
        
        """
        super().__init__(epochs, loss_function, validation, early_stopping, optimizer, optimizer_params)
        
    
    def _init_optimizer(self, model, optimizer, optim_params):
        """
        Inicializa el optimizador a utilizar en el entrenamiento, utilizando los parámetros del modelo y los parámetros del optimizador especificados.
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
            optimizer (torch.optim): Optimizador a utilizar en el entrenamiento.
            optim_params (dict): Parámetros a utilizar en el optimizador.
            
        Returns:
            torch.optim.Optimizer: Instancia del optimizador.
        """
        return optimizer(model.parameters(), **optim_params)
    
    
    def _update_parameters(self, model, loader):
        """
        Actualiza los parámetros del modelo (época única).
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
            loader (DataLoader): DataLoader con los datos de entrenamiento.
        """
        optimizer_training_epoch(model, loader, self._optimizer_instance, self.loss_function)