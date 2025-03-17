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
        No debe ser instanciada directamente. Solo sirve para ser heredada por otras clases. 
        
    Note:
        Las clases que hereden de esta clase deben implementar las funciones:
            - **_update_parameters(model, loader)**: ejecuta la actualización de parámetros durante una sola época el entrenamiento del modelo.
            - **init_optimizer(self, model, optimizer, optim_params)**: instancia el optimizador a utilizar durante el entrenamiento del modelo.
        
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
        train_loader, val_loader = self._train_val_split(model, loader)
        
        self._init_optimizer(model)
        
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
        if self.loss_function != torch.nn.functional.cross_entropy: #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            if y.dtype != pred.dtype:
                y = y.to(pred.dtype)
        else: 
            y = y.type(torch.int64)
        '''preliminary fix for the dtype issue'''
        
        return self.loss_function(pred, y)
    
    def _train_val_split(self, model, train_loader):
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
            
            if model._output_type == 'regression':
                x_train, x_val, y_train, y_val = train_test_split(x_train.numpy(), y_train.numpy(), test_size=self.validation, shuffle=True)
            else:
                x_train, x_val, y_train, y_val = train_test_split(x_train.numpy(), y_train.numpy(), test_size=self.validation, shuffle=True, stratify=y_train.numpy())
                
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
        if self.validation > 0 and val_loader is not None:
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
    
    Note:
        Este algoritmo está basado en el algoritmo de entrenamiento híbrido propuesto por Jang en 1993, para mas información ver: `ANFIS: adaptive-network-based fuzzy inference system <https://doi.org/10.1109/21.256541>`_.
    
    Pseudocódigo:
    
    .. image:: ../../_static/Hybrid_learning_algorithm_pseudocode.png
        :alt: Pseudocódigo del algoritmo de entrenamiento híbrido.
        :align: center
        :width: 600px
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
        
        
    def _init_optimizer(self, model):
        """
        Inicializa el optimizador a utilizar en el entrenamiento, utilizando los parámetros de los antececentes del modelo ANFIS y los parámetros del optimizador especificados.
        
        Args:
            model (ANFIS | h_ANFIS): Modelo ANFIS a entrenar.
        """
        self._optimizer_instance = self.optimizer(model.get_premises_as_parameters_list(), **self.optimizer_params)

    
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
        
        
    def _sonfis_init_optimizer(self, model, freezed_subnets):
        """
        Inicializa el optimizador a utilizar en el entrenamiento, utilizando algunos los parámetros de los antececentes del modelo ANFIS y los parámetros del optimizador especificados.
        
        Args:
            model (h_ANFIS [rule_reduced]): Modelo h_ANFIS que debe ser rule_reduced.
            freezed_subnets (torch.tensor): Tensor booleano que indica que antecedentes no deben ser actualizados.
        """
        premises_parameters_to_train = []
        
        ###########
        ###########
        ###########
        #freezed_subnets = torch.zeros_like(freezed_subnets, dtype=torch.bool)
        ###########
        ###########
        ###########
        
        not_freezed = torch.where(~freezed_subnets)[0]
        for i in not_freezed:
            premises_parameters_to_train.append(model.get_premises_as_parameters_list()[i.item()])
            
        if not_freezed.size(0) == 0:
            self._optimizer_instance = None
        else:
            self._optimizer_instance = self.optimizer(premises_parameters_to_train, **self.optimizer_params)
        
        
    def _sonfis_update_parameters(self, model, train_loader, val_loader, freezed_subnets):
        """
        Aplica el algoritmo de entrenamiento de un modelo h_ANFIS dado un conjunto de subredes que no se deben actualizar.
        
        Args:
            model (h_ANFIS [rule_reduced]): Modelo h_ANFIS que debe ser rule_reduced.
            train_loader (DataLoader): DataLoader con los datos de entrenamiento.
            val_loader (DataLoader): DataLoader con los datos de validación.
            freezed_subnets (torch.tensor): Tensor booleano que indica que subredes no deben ser actualizadas.
        """
        
        self._sonfis_init_optimizer(model, freezed_subnets)
        
        # unfrozen parameters used to instantiate the optimizer to be empty (which is not allowed in PyTorch).
        if self._optimizer_instance is None: # There is a possibility that no subnets are created or divided, but some may vanish. This would cause the list of parameters to
            return                           # be empty, which is not allowed in PyTorch.
        

        current_consequents = model.get_consequents()
        
        ep = 0
        while ep < self.epochs:
            self._update_parameters(model, train_loader)
            
            if self.validation > 0 and self.early_stopping is not None:
                _, val_loss = self._loss(model, train_loader, val_loader)
                self.early_stopping(model, val_loss, False)
                if self.early_stopping._stop:
                    break
                
            #loss, val_loss = self._loss(model, train_loader, val_loader)
            #epoch_width = len(str(self.epochs))
            #if self.validation > 0:
            #    print(f'Epoch: {ep+1:{epoch_width}}/{self.epochs} - loss: {loss:.6f} - validation loss: {val_loss:.6f}')
            #else:
            #    print(f'Epoch: {ep+1:{epoch_width}}/{self.epochs} - loss: {loss:.6f}')
                
            ep += 1
        
        new_consequents = model.get_consequents()
        new_consequents[:, freezed_subnets, :] = current_consequents[:, freezed_subnets, :]
        model.set_consequents(new_consequents)
        
        if self.early_stopping is not None:
            self.early_stopping.reset()
            
        #print(ep)
        """
        
        current_consequents = model.get_consequents()
        current_premises = model.get_premises()
        #print(model.get_premises())
        ep = 0
        #self._optimizer_instance = self._init_optimizer(model, self.optimizer, self.optimizer_params)
        while ep < self.epochs:
            #---------------------Consequents update------------------------
            ### current_consequents = ANFISmodel.get_consequents()
            
            model.set_consequents(classical_consequents_estimation_with_OLS(model, train_loader))
            ### new_consequents = ANFISmodel.get_consequents()
            
            ### freezed_consequents = self._freezed
            
            ### new_consequents[:, freezed_consequents, :] = current_consequents[:, freezed_consequents, :]
            ### ANFISmodel.set_consequents(new_consequents)
            
            # ------------------Premises update-----------------------
            ### current_premises = ANFISmodel.get_premises()
            optimizer_training_epoch(model, train_loader, self._optimizer_instance, self.loss_function)
            ### new_premises = ANFISmodel.get_premises()
            #print(model.get_premises())
            ### freezed_premises = self._freezed
            
            ### new_premises[:, freezed_premises, :] = current_premises[:, freezed_premises, :]
            ### ANFISmodel.set_premises(new_premises)
            
            #loss, val_loss = self._loss(ANFISmodel, train_loader, val_loader)
            #epoch_width = len(str(self.trainer_epochs))
            #if self.validation > 0:
            #    print(f'    Epoch: {ep+1:{epoch_width}}/{self.trainer_epochs} - loss: {loss:.6f} - validation loss: {val_loss:.6f}')
            #else:
            #    print(f'    Epoch: {ep+1:{epoch_width}}/{self.trainer_epochs} - loss: {loss:.6f}')
            
            if self.validation > 0 and self.early_stopping is not None:
                _, val_loss = self._loss(model, train_loader, val_loader)
                self.early_stopping(model, val_loss, False)
                if self.early_stopping._stop:
                    break
                
            ep += 1
            
        new_consequents = model.get_consequents()
        new_consequents[:, freezed_subnets, :] = current_consequents[:, freezed_subnets, :]
        model.set_consequents(new_consequents)
        
        new_premises = model.get_premises()
        new_premises[:, freezed_subnets, :] = current_premises[:, freezed_subnets, :]
        model.set_premises(new_premises)
        
        if self.early_stopping is not None:
            self.early_stopping.reset()     
        """


class Basic_optimizer_training_algorithm(base_model_trainer):
    """
    Clase para representar un algoritmo de entrenamiento por optimización para un modelo de aprendizaje automático. Solo permite entrenar modelos de aprendizaje automático con un solo optimizador para todos los parámetros del modelo.
    
    Pseudocódigo:
    
    .. image:: ../../_static/Basic_optimizer_training_algorithm_pseudocode.png
        :alt: Pseudocódigo del algorithmo de entrenamiento básico por optimización.
        :align: center
        :width: 600px
    """
    
    def __init__(self, epochs, loss_function, validation=0, early_stopping=None, optimizer=torch.optim.Adam, optimizer_params={}):
        """
        Inicializa una nueva instancia de la clase Basic_optimizer_training_algorithm.
        
        Args:
            epochs (int): Número de épocas de entrenamiento.
            loss_function (torch.nn.functional): Función de pérdida a utilizar en el entrenamiento.
            validation (float): Proporción de los datos de entrenamiento a utilizar como datos de validación (Default: 0).
            early_stopping (EarlyStopping): Mecanismo de Early Stopping a utilizar en el entrenamiento (Default: None).
            optimizer (torch.optim): Optimizador a utilizar en el entrenamiento (Default: torch.optim.Adam).
            optimizer_params (dict): Parámetros a utilizar en el optimizador (Default: {}).
        
        """
        super().__init__(epochs, loss_function, validation, early_stopping, optimizer, optimizer_params)
        
    
    def _init_optimizer(self, model):
        """
        Inicializa el optimizador a utilizar en el entrenamiento, utilizando los parámetros del modelo y los parámetros del optimizador especificados.
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
        """
        self._optimizer_instance = self.optimizer(model.parameters(), **self.optimizer_params)
    
    
    def _update_parameters(self, model, loader):
        """
        Actualiza los parámetros del modelo (época única).
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
            loader (DataLoader): DataLoader con los datos de entrenamiento.
        """
        optimizer_training_epoch(model, loader, self._optimizer_instance, self.loss_function)
        
        
    def _sonfis_init_optimizer(self, model, freezed_subnets):
        """
        Inicializa el optimizador a utilizar en el entrenamiento, utilizando algunos los parámetros de los antececentes del modelo ANFIS y los parámetros del optimizador especificados.
        
        Args:
            model (h_ANFIS [rule_reduced]): Modelo h_ANFIS que debe ser rule_reduced.
            freezed_subnets (torch.tensor): Tensor booleano que indica que antecedentes no deben ser actualizados.
        """
        parameters_to_train = []
        not_freezed = torch.where(~freezed_subnets)[0]
        for i in not_freezed:
            parameters_to_train.append(model.get_premises_as_parameters_list()[i.item()])
            parameters_to_train.append(model.get_consequents_as_parameters_list()[i.item()])
            
        if not_freezed.size(0) == 0:
            self._optimizer_instance = None
        else:
            self._optimizer_instance = self.optimizer(parameters_to_train, **self.optimizer_params)
          
            
    def _sonfis_update_parameters(self, model, train_loader, val_loader, freezed_subnets):
        """
        Aplica el algoritmo de entrenamiento de un modelo h_ANFIS dado un conjunto de subredes que no se deben actualizar.
        
        Args:
            model (h_ANFIS [rule_reduced]): Modelo h_ANFIS que debe ser rule_reduced.
            train_loader (DataLoader): DataLoader con los datos de entrenamiento.
            val_loader (DataLoader): DataLoader con los datos de validación.
            freezed_subnets (torch.tensor): Tensor booleano que indica que subredes no deben ser actualizadas.
        """
        self._sonfis_init_optimizer(model, freezed_subnets)
        
        # unfrozen parameters used to instantiate the optimizer to be empty (which is not allowed in PyTorch).
        if self._optimizer_instance is None:
            return
        
        ep = 0
        while ep < self.epochs:
            self._update_parameters(model, train_loader)
            
            if self.validation > 0 and self.early_stopping is not None:
                _, val_loss = self._loss(model, train_loader, val_loader)
                self.early_stopping(model, val_loss, False)
                if self.early_stopping._stop:
                    break
                
            ep += 1
            
        if self.early_stopping is not None:
            self.early_stopping.reset()
        
        

class Double_optimizer_training_algorithm(base_model_trainer):
    """
    Clase para representar un algoritmo de entrenamiento por optimización para un modelo de Sistema de Inferencia Neuro-Difuso Adaptativo (ANFIS).
    La idea es que los antecedentes y los consecuentes del modelo ANFIS se actualicen con optimizadores diferentes.
    
    Pseudocódigo:
    
    HAY QUE HACER EL PSEUDOCÓDIGO
    HAY QUE HACER EL PSEUDOCÓDIGO
    HAY QUE HACER EL PSEUDOCÓDIGO
    HAY QUE HACER EL PSEUDOCÓDIGO
    
    .. image:: ../../_static/Basic_optimizer_training_algorithm_pseudocode.png
        :alt: Pseudocódigo del algorithmo de entrenamiento básico por optimización.
        :align: center
        :width: 600px
        
    HAY QUE HACER EL PSEUDOCÓDIGO
    HAY QUE HACER EL PSEUDOCÓDIGO
    HAY QUE HACER EL PSEUDOCÓDIGO
    HAY QUE HACER EL PSEUDOCÓDIGO
    
    """
    
    def __init__(self, epochs, loss_function, validation=0, early_stopping=None, prems_optim=torch.optim.Adam, prems_optim_params={}, cons_optim=torch.optim.Adam, cons_optim_params={}):
        """
        Inicializa una nueva instancia de la clase Basic_optimizer_training_algorithm.
        
        Args:
            epochs (int): Número de épocas de entrenamiento.
            loss_function (torch.nn.functional): Función de pérdida a utilizar en el entrenamiento.
            validation (float): Proporción de los datos de entrenamiento a utilizar como datos de validación (Default: 0).
            early_stopping (EarlyStopping): Mecanismo de Early Stopping a utilizar en el entrenamiento (Default: None).
            optimizer1 (torch.optim): Optimizador a utilizar en el entrenamiento (Default: torch.optim.Adam).
            optimizer_params1 (dict): Parámetros a utilizar en el optimizador (Default: {}).
        
        """
        super().__init__(epochs, loss_function, validation, early_stopping, optimizer=None, optimizer_params={})
        self.prems_optim = prems_optim
        self.prems_optim_params = prems_optim_params
        self.cons_optim = cons_optim
        self.cons_optim_params = cons_optim_params
        
        self._prems_optimizer_instance = None
        self._cons_optimizer_instance = None
        
    
    def _init_optimizer(self, model):
        """
        Inicializa el optimizador a utilizar en el entrenamiento, utilizando los parámetros del modelo y los parámetros del optimizador especificados.
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
        """
        self._prems_optimizer_instance = self.prems_optim(model.get_premises_as_parameters_list(), **self.prems_optim_params)
        self._cons_optimizer_instance = self.cons_optim(model.get_consequents_as_parameters_list(), **self.cons_optim_params)
    
    
    def _update_parameters(self, model, loader):
        """
        Actualiza los parámetros del modelo (época única).
        
        Args:
            model (torch.nn.Module): Modelo a entrenar.
            loader (DataLoader): DataLoader con los datos de entrenamiento.
        """
        for batch_x, batch_y in loader:
            batch_y_copy = batch_y.clone().detach()

            '''preliminary fix for the dtype issue'''
            if self.loss_function != torch.nn.functional.cross_entropy:
                if loader.dataset.tensors[0].dtype != loader.dataset.tensors[1].dtype:
                    batch_y_copy = batch_y_copy.to(batch_x.dtype)
            else: 
                batch_y_copy = batch_y_copy.to(torch.int64) #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            '''preliminary fix for the dtype issue'''

            self._prems_optimizer_instance.zero_grad()
            self._cons_optimizer_instance.zero_grad()
            pred = model(batch_x)
            loss = self.loss_function(pred, batch_y_copy)
            loss.backward()
            self._prems_optimizer_instance.step()
            self._cons_optimizer_instance.step()

            if torch.isnan(loss):
                print("--- prem grads --- prem grads --- prem grads ----")
                for i in range (model._input_size):
                    print(model._fuzzification_layer._premises[i].grad)

                print("")
                print("--- prem param --- prem param --- prem param ----")
                print(model.premises_structure)
                print("")

                raise ValueError('Loss is NaN')
            
    
    def _sonfis_init_optimizer(self, model, freezed_subnets):
        """
        Inicializa el optimizador a utilizar en el entrenamiento, utilizando algunos los parámetros de los antececentes del modelo ANFIS y los parámetros del optimizador especificados.
        
        Args:
            model (h_ANFIS [rule_reduced]): Modelo h_ANFIS que debe ser rule_reduced.
            freezed_subnets (torch.tensor): Tensor booleano que indica que antecedentes no deben ser actualizados.
        """
        parameters_to_train = []
        not_freezed = torch.where(~freezed_subnets)[0]
        for i in not_freezed:
            parameters_to_train.append(model.get_premises_as_parameters_list()[i.item()])
            parameters_to_train.append(model.get_consequents_as_parameters_list()[i.item()])
            
        if not_freezed.size(0) == 0:
            self._prems_optimizer_instance = None
            self._cons_optimizer_instance = None
        else:
            self._prems_optimizer_instance = self.prems_optim(parameters_to_train, **self.prems_optim_params)
            self._cons_optimizer_instance = self.cons_optim(parameters_to_train, **self.cons_optim_params)
            
    
    def _sonfis_update_parameters(self, model, train_loader, val_loader, freezed_subnets):
        """
        Aplica el algoritmo de entrenamiento de un modelo h_ANFIS dado un conjunto de subredes que no se deben actualizar.
        
        Args:
            model (h_ANFIS [rule_reduced]): Modelo h_ANFIS que debe ser rule_reduced.
            train_loader (DataLoader): DataLoader con los datos de entrenamiento.
            val_loader (DataLoader): DataLoader con los datos de validación.
            freezed_subnets (torch.tensor): Tensor booleano que indica que subredes no deben ser actualizadas.
        """
        self._sonfis_init_optimizer(model, freezed_subnets)
        
        # unfrozen parameters used to instantiate the optimizer to be empty (which is not allowed in PyTorch).
        if self._prems_optimizer_instance is None:
            return
        
        ep = 0
        while ep < self.epochs:
            self._update_parameters(model, train_loader)
            
            if self.validation > 0 and self.early_stopping is not None:
                _, val_loss = self._loss(model, train_loader, val_loader)
                self.early_stopping(model, val_loss, False)
                if self.early_stopping._stop:
                    break
                
            ep += 1
            
        if self.early_stopping is not None:
            self.early_stopping.reset()