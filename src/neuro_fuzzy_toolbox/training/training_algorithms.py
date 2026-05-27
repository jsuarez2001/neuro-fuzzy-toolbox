import torch
import torch.nn as nn

from neuro_fuzzy_toolbox.training import (
    classical_consequents_estimation_with_OLS,
    optimizer_training_epoch
)

class base_model_trainer():
    """
    Base class for representing a training algorithm for a machine learning model.

    Warning:
        This class should not be instantiated directly. It is intended to be
        subclassed by other classes.

    Note:
        Subclasses must implement the following methods:

        - **_update_parameters(model, loader)**: executes the parameter update during a single training epoch.
        - **init_optimizer(self, model, optimizer, optim_params)**: instantiates the optimizer to be used during training.
        
        To be compatible with the SONFIS algorithm, subclasses must additionally implement:

        - **_sonfis_update_parameters(model, train_loader, val_loader, freezed_subnets)**: applies the training algorithm within the SONFIS procedure.
        - **_sonfis_init_optimizer(model, freezed_subnets)**: initializes the optimizer for use within the SONFIS procedure.
    """
    
    def __init__(self, epochs, loss_function, early_stopping=None, optimizer=torch.optim.Adam, optimizer_params={}):
        """
        Initializes a new base_model_trainer instance.

        Args:
            epochs (int): Number of training epochs.
            loss_function (torch.nn.Module): Instantiated loss function to use during training.
            early_stopping (EarlyStopping): Early stopping mechanism to use during training. Defaults to ``None``.
            optimizer (torch.optim.Optimizer): Optimizer class to use during training. Defaults to ``torch.optim.Adam``.
            optimizer_params (dict): Parameters to pass to the optimizer. Defaults to ``{}``.
        """
        # Training parameters
        self.epochs = epochs
        self.loss_function = loss_function
        
        # Optimizer
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self._optimizer_instance = None
        
        # Early stopping
        self.early_stopping = early_stopping
        
        # History
        self.history = {"loss": []}
        self.val_history = {"loss": []}
        

        
    def __call__(self, model, train_loader, val_loader=None, verbose=True):
        """
        Trains a model using the training algorithm.

        Args:
            model (torch.nn.Module): Model to train.
            train_loader (DataLoader): DataLoader containing the training data.
            val_loader (DataLoader): DataLoader containing the validation data. Defaults to ``None``.
            verbose (bool): If ``True``, prints progress messages during training. Defaults to ``True``.
        """
        self._init_optimizer(model)
        
        self._register_loss(model, train_loader, val_loader)

        ep = 0
        while ep < self.epochs:
            
            self._update_parameters(model, train_loader)
            
            self._register_loss(model, train_loader, val_loader)
            
            loss, val_loss = self.history["loss"][-1], None
            
            if (val_loader is not None):
                val_loss = self.val_history["loss"][-1]
                if self._check_early_stop(model, val_loss, verbose):
                    break
                
            if verbose:
                epoch_width = len(str(self.epochs))
                if not (val_loader is None):
                    print(f'Epoch: {ep+1:{epoch_width}}/{self.epochs} - loss: {loss:.6f} - validation loss: {val_loss:.6f}')
                else:
                    print(f'Epoch: {ep+1:{epoch_width}}/{self.epochs} - loss: {loss:.6f}')
                    
            ep += 1
        
        print('Training finished')
            
            
    def _loss_function(self, model, pred, y):
        """
        Computes the loss value.

        Args:
            model (torch.nn.Module): Model being trained.
            pred (torch.Tensor): Tensor containing the model predictions.
            y (torch.Tensor): Tensor containing the ground truth labels.

        Returns:
            torch.Tensor: Tensor containing the loss value.
        """
        
        '''preliminary fix for the dtype issue'''
        if not isinstance(self.loss_function, nn.CrossEntropyLoss): #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            if y.dtype != pred.dtype:
                y = y.to(pred.dtype)
        else: 
            y = y.type(torch.int64)
        '''preliminary fix for the dtype issue'''
        
        if isinstance(self.loss_function, nn.CrossEntropyLoss) and model._custom_classes:
            y = torch.searchsorted(model.classes, y).long()
        
        return self.loss_function(pred, y)
    
    
    def _check_early_stop(self, model, loss, verbose):
        """
        Checks whether training should be stopped via the early stopping mechanism, if one is set.

        Args:
            model (torch.nn.Module): Model being trained.
            loss (float): Current loss value of the model.
            verbose (bool): If ``True``, prints a warning message when early stopping is triggered.

        Returns:
            bool: ``True`` if training should be stopped, ``False`` otherwise.
        """
        if self.early_stopping is not None:
            self.early_stopping(model, loss, verbose)
            if self.early_stopping.stop:
                self.early_stopping.reset()
                return True
        return False
    
    
    def _loss(self, model, train_loader, val_loader):
        """
        Computes the training and validation loss values of the model.

        Args:
            model (torch.nn.Module): Model being trained.
            train_loader (DataLoader): DataLoader containing the training data.
            val_loader (DataLoader): DataLoader containing the validation data.

        Returns:
            torch.Tensor: Tensor containing the training loss value.
            torch.Tensor: Tensor containing the validation loss value, or ``None`` if no validation data is provided.
        """
        val_loss = None
        x = train_loader.dataset.tensors[0]
        y = train_loader.dataset.tensors[1]
        with torch.no_grad():
            pred = model(x)
            loss = self._loss_function(model, pred, y)
        if val_loader is not None:
            x = val_loader.dataset.tensors[0]
            y = val_loader.dataset.tensors[1]
            with torch.no_grad():
                pred = model(x)
                val_loss = self._loss_function(model, pred, y)
        return loss, val_loss
    
    
    def _register_loss(self, model, train_loader, val_loader):
        """
        Records the current loss value in the training history. If a validation
        DataLoader is provided, also records the validation loss.

        Args:
            model (torch.nn.Module): Model being trained.
            train_loader (DataLoader): DataLoader containing the training data.
            val_loader (DataLoader): DataLoader containing the validation data, or ``None`` if no validation data is used.
        """
        loss, val_loss = self._loss(model, train_loader, val_loader)
        self.history["loss"].append(loss.item())
        if not (val_loader is None):
            self.val_history["loss"].append(val_loss.item())



class Hybrid_learning_algorithm(base_model_trainer):
    """
    Hybrid learning algorithm for training an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.
    
    Note:
        This algorithm is based on the hybrid training algorithm proposed by Jang in 1993. For more information, see:
        `ANFIS: adaptive-network-based fuzzy inference system <https://doi.org/10.1109/21.256541>`_.

    Pseudocode:
    
    .. image:: ../../_static/Hybrid_learning_algorithm_pseudocode.png
        :alt: Pseudocode of the hybrid learning algorithm.
        :align: center
        :width: 600px
    """
    def __init__(self, epochs, loss_function, driver=None, ridge_lambda=0., early_stopping=None, optimizer=torch.optim.Adam, optimizer_params={}):
        """
        Initializes a new Hybrid_learning_algorithm instance.

        Args:
            epochs (int): Number of training epochs.
            loss_function (torch.nn.Module): Instantiated loss function to use during training.
            driver (str): Backend function to use for the least-squares estimation of the consequent parameters. 
                Valid values are ``'gels'``, ``'gelsy'``, ``'gelsd'``, and ``'gelss'``. If ``None``, defaults to ``'gels'``.
            ridge_lambda (float): Lambda value for Ridge regularization in the least-squares estimation. 
                If ``0.``, no regularization is applied. Defaults to ``0.``.
            early_stopping (EarlyStopping): Early stopping mechanism to use during training. Defaults to ``None``.
            optimizer (torch.optim.Optimizer): Optimizer class to use during training. Defaults to ``torch.optim.Adam``.
            optimizer_params (dict): Parameters to pass to the optimizer. Defaults to ``{}``.
        """
        super().__init__(epochs, loss_function, early_stopping, optimizer, optimizer_params)
        self.driver = driver
        self.ridge_lambda = ridge_lambda
        
        
    def _init_optimizer(self, model):
        """
        Initializes the optimizer using only the premise parameters of the ANFIS model and the specified optimizer parameters.

        Args:
            model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
        """
        self._optimizer_instance = self.optimizer(model.get_premises_as_parameters_list(), **self.optimizer_params)

    
    def _premises_update(self, ANFISmodel, loader):
        """
        Updates the premise parameters of the ANFIS model for one training epoch.

        Args:
            ANFISmodel (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
            loader (DataLoader): DataLoader containing the training data.
        """
        optimizer_training_epoch(ANFISmodel, loader, self._optimizer_instance, self.loss_function)
        
    
    def _consequents_update(self, ANFISmodel, loader):
        """
        Updates the consequent parameters of the ANFIS model using ordinary least squares estimation.

        Args:
            ANFISmodel (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
            loader (DataLoader): DataLoader containing the training data.
        """
        ANFISmodel.set_consequents(classical_consequents_estimation_with_OLS(ANFISmodel, loader, self.driver, self.ridge_lambda))
    

    def _update_parameters(self, ANFISmodel, loader):
        """
        Updates both the consequent and premise parameters of the ANFIS model for a single training epoch. 
        The consequent update is performed first, followed by the premise update.

        Args:
            ANFISmodel (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
            loader (DataLoader): DataLoader containing the training data.
        """
        self._consequents_update(ANFISmodel, loader)
        self._premises_update(ANFISmodel, loader)
        
        
    def _sonfis_init_optimizer(self, model, freezed_subnets):
        """
        Initializes the optimizer for use within the SONFIS training algorithm, using only the premise parameters 
        of the subnets that are not frozen.

        Note:
            This method is intended exclusively for use within the SONFIS algorithm and should not be called directly in standard training.

        Args:
            model (rule_reduced_ANFIS): Rule-reduced ANFIS model to train.
            freezed_subnets (torch.Tensor): Boolean tensor indicating which subnets should not be updated.
        """
        premises_parameters_to_train = []
        
        not_freezed = torch.where(~freezed_subnets)[0]
        for i in not_freezed:
            premises_parameters_to_train.append(model.get_premises_as_parameters_list()[i.item()])
            
        if not_freezed.size(0) == 0:
            self._optimizer_instance = None
        else:
            self._optimizer_instance = self.optimizer(premises_parameters_to_train, **self.optimizer_params)
        
        
    def _sonfis_update_parameters(self, model, train_loader, val_loader, freezed_subnets):
        """
        Applies the hybrid learning algorithm within the SONFIS training procedure, updating only the subnets 
        that are not frozen. After training, the consequent parameters of the frozen subnets are restored
        to their values prior to the update.
        
        Note:
            This method is intended exclusively for use within the SONFIS
            algorithm and should not be called directly in standard training.
        
        Args:
            model (rule_reduced_ANFIS): Rule-reduced ANFIS model to train.
            train_loader (DataLoader): DataLoader containing the training data.
            val_loader (DataLoader): DataLoader containing the validation data, or ``None`` if no validation data is used.
            freezed_subnets (torch.Tensor): Boolean tensor indicating which subnets should not be updated.
        """
        
        self._sonfis_init_optimizer(model, freezed_subnets)
        
        if self._optimizer_instance is None: # There is a possibility that no subnets are created or divided, but some may vanish. 
            return                           # This would cause the list of parameters to be empty, which is not allowed in PyTorch.
        
        current_consequents = model.get_consequents()
        
        ep = 0
        while ep < self.epochs:
            self._update_parameters(model, train_loader)
            
            if (val_loader is not None) and (self.early_stopping is not None):
                _, val_loss = self._loss(model, train_loader, val_loader)
                self.early_stopping(model, val_loss, False)
                if self.early_stopping._stop:
                    break
                
            ep += 1
        
        new_consequents = model.get_consequents()
        new_consequents[:, freezed_subnets, :] = current_consequents[:, freezed_subnets, :]
        model.set_consequents(new_consequents)
        
        if self.early_stopping is not None:
            self.early_stopping.reset()


class Basic_optimizer_training_algorithm(base_model_trainer):
    """
    Optimizer-based training algorithm for a machine learning model. Trains all model parameters using a single optimizer.

    Pseudocode:

    .. image:: ../../_static/Basic_optimizer_training_algorithm_pseudocode.png
        :alt: Pseudocode of the basic optimizer-based training algorithm.
        :align: center
        :width: 600px
    """
    
    def __init__(self, epochs, loss_function, early_stopping=None, optimizer=torch.optim.Adam, optimizer_params={}):
        """
        Initializes a new Basic_optimizer_training_algorithm instance.

        Args:
            epochs (int): Number of training epochs.
            loss_function (torch.nn.Module): Instantiated loss function to use during training.
            early_stopping (EarlyStopping): Early stopping mechanism to use during training. Defaults to ``None``.
            optimizer (torch.optim.Optimizer): Optimizer class to use during training. Defaults to ``torch.optim.Adam``.
            optimizer_params (dict): Parameters to pass to the optimizer. Defaults to ``{}``.
        """
        super().__init__(epochs, loss_function, early_stopping, optimizer, optimizer_params)
        
    
    def _init_optimizer(self, model):
        """
        Initializes the optimizer using all parameters of the model and the specified optimizer parameters.

        Args:
            model (torch.nn.Module): Model to train.
        """
        self._optimizer_instance = self.optimizer(model.parameters(), **self.optimizer_params)
    
    
    def _update_parameters(self, model, loader):
        """
        Updates all model parameters for a single training epoch.

        Args:
            model (torch.nn.Module): Model to train.
            loader (DataLoader): DataLoader containing the training data.
        """
        optimizer_training_epoch(model, loader, self._optimizer_instance, self.loss_function)
        
        
    def _sonfis_init_optimizer(self, model, freezed_subnets):
        """
        Initializes the optimizer for use within the SONFIS training algorithm,
        using both the premise and consequent parameters of the subnets that
        are not frozen.
        
        Note:
            This method is intended exclusively for use within the SONFIS
            algorithm and should not be called directly in standard training.
        
        Args:
            model (rule_reduced_ANFIS): Rule-reduced ANFIS model to train.
            freezed_subnets (torch.Tensor): Boolean tensor indicating which subnets should not be updated.
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
        Applies the optimizer-based training algorithm within the SONFIS
        training procedure, updating only the subnets that are not frozen.
        
        Note:
            This method is intended exclusively for use within the SONFIS
            algorithm and should not be called directly in standard training.
        
        Args:
            model (rule_reduced_ANFIS): Rule-reduced ANFIS model to train.
            train_loader (DataLoader): DataLoader containing the training data.
            val_loader (DataLoader): DataLoader containing the validation data, or ``None`` if no validation data is used.
            freezed_subnets (torch.Tensor): Boolean tensor indicating which subnets should not be updated.
        """
        self._sonfis_init_optimizer(model, freezed_subnets)
        
        # unfrozen parameters used to instantiate the optimizer to be empty (which is not allowed in PyTorch).
        if self._optimizer_instance is None:
            return
        
        ep = 0
        while ep < self.epochs:
            self._update_parameters(model, train_loader)
            
            if (val_loader is not None) and (self.early_stopping is not None):
                _, val_loss = self._loss(model, train_loader, val_loader)
                self.early_stopping(model, val_loss, False)
                if self.early_stopping._stop:
                    break
                
            ep += 1
            
        if self.early_stopping is not None:
            self.early_stopping.reset()
        
        

class Double_optimizer_training_algorithm(base_model_trainer):
    """
    Dual-optimizer training algorithm for an Adaptive Neuro-Fuzzy Inference
    System (ANFIS) model. Uses separate optimizers to update the premise and
    consequent parameters independently.
    
    The update strategy is controlled by the ``mode`` parameter:
    
    - **Mode 0**: In each epoch, a single forward pass is performed and both
      the premise and consequent parameters are updated simultaneously using
      their respective optimizers.
    - **Mode 1**: Each epoch consists of two sequential passes through the
      data. In the first pass, only the consequent parameters are updated;
      in the second pass, only the premise parameters are updated.
    """
    
    def __init__(self, epochs, loss_function, early_stopping=None, mode=0, prems_optim=torch.optim.Adam, prems_optim_params={}, cons_optim=torch.optim.Adam, cons_optim_params={}):
        """
        Initializes a new Double_optimizer_training_algorithm instance.
        
        Args:
            epochs (int): Number of training epochs.
            loss_function (torch.nn.Module): Instantiated loss function to use during training.
            early_stopping (EarlyStopping): Early stopping mechanism to use during training. Defaults to ``None``.
            mode (int): Update strategy to use. ``0`` updates premises and consequents simultaneously in a single pass; 
                ``1`` updates them sequentially in two separate passes per epoch. Defaults to ``0``.
            prems_optim (torch.optim.Optimizer): Optimizer class to use for the premise parameters. Defaults to ``torch.optim.Adam``.
            prems_optim_params (dict): Parameters to pass to the premise optimizer. Defaults to ``{}``.
            cons_optim (torch.optim.Optimizer): Optimizer class to use for the consequent parameters. Defaults to ``torch.optim.Adam``.
            cons_optim_params (dict): Parameters to pass to the consequent optimizer. Defaults to ``{}``.
        """
        super().__init__(epochs, loss_function, early_stopping, optimizer=None, optimizer_params={})
        self.prems_optim = prems_optim
        self.prems_optim_params = prems_optim_params
        self.cons_optim = cons_optim
        self.cons_optim_params = cons_optim_params
        
        self._prems_optimizer_instance = None
        self._cons_optimizer_instance = None
        
        if mode == 0:
            self._update_parameters = self._update_parameters_0
        else:
            self._update_parameters = self._update_parameters_1
        
    
    def _init_optimizer(self, model):
        """
        Initializes both optimizers using the premise and consequent parameters of the model, respectively,
        along with the specified optimizer parameters.
        
        Args:
            model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
        """
        self._prems_optimizer_instance = self.prems_optim(model.get_premises_as_parameters_list(), **self.prems_optim_params)
        self._cons_optimizer_instance = self.cons_optim(model.get_consequents_as_parameters_list(), **self.cons_optim_params)
    
    
    def _update_parameters_0(self, model, loader):
        """
        Updates both the premise and consequent parameters simultaneously for
        a single training epoch, using a single forward pass per batch.
        
        Args:
            model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
            loader (DataLoader): DataLoader containing the training data.
        """
        for batch_x, batch_y in loader:
            batch_y_copy = batch_y.clone().detach()

            '''preliminary fix for the dtype issue'''
            if not isinstance(self.loss_function, nn.CrossEntropyLoss):
                if loader.dataset.tensors[0].dtype != loader.dataset.tensors[1].dtype:
                    batch_y_copy = batch_y_copy.to(batch_x.dtype)
            else: 
                batch_y_copy = batch_y_copy.to(torch.int64) #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            '''preliminary fix for the dtype issue'''
            
            if isinstance(self.loss_function, nn.CrossEntropyLoss) and model._custom_classes:
                batch_y_copy = torch.searchsorted(model.classes, batch_y_copy).long()

            self._prems_optimizer_instance.zero_grad()
            self._cons_optimizer_instance.zero_grad()
            pred = model(batch_x)
            loss = self.loss_function(pred, batch_y_copy)
            loss.backward()
            self._prems_optimizer_instance.step()
            self._cons_optimizer_instance.step()

            if torch.isnan(loss):
                raise ValueError('Loss is NaN')
            
    def _update_parameters_1(self, model, loader):
        """
        Updates the premise and consequent parameters sequentially for a single training epoch, using two separate passes
        through the data. In the first pass, only the consequent parameters are updated; in the second pass, only the 
        premise parameters are updated.

        Args:
            model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
            loader (DataLoader): DataLoader containing the training data.
        """
        for batch_x, batch_y in loader:
            batch_y_copy = batch_y.clone().detach()

            '''preliminary fix for the dtype issue'''
            if not isinstance(self.loss_function, nn.CrossEntropyLoss):
                if loader.dataset.tensors[0].dtype != loader.dataset.tensors[1].dtype:
                    batch_y_copy = batch_y_copy.to(batch_x.dtype)
            else: 
                batch_y_copy = batch_y_copy.to(torch.int64) #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            '''preliminary fix for the dtype issue'''
            
            if isinstance(self.loss_function, nn.CrossEntropyLoss) and model._custom_classes:
                batch_y_copy = torch.searchsorted(model.classes, batch_y_copy).long()

            self._cons_optimizer_instance.zero_grad()
            pred = model(batch_x)
            loss = self.loss_function(pred, batch_y_copy)
            loss.backward()
            self._cons_optimizer_instance.step()

            if torch.isnan(loss):
                raise ValueError('Loss is NaN')
            
        for batch_x, batch_y in loader:
            batch_y_copy = batch_y.clone().detach()

            '''preliminary fix for the dtype issue'''
            if not isinstance(self.loss_function, nn.CrossEntropyLoss):
                if loader.dataset.tensors[0].dtype != loader.dataset.tensors[1].dtype:
                    batch_y_copy = batch_y_copy.to(batch_x.dtype)
            else: 
                batch_y_copy = batch_y_copy.to(torch.int64) #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            '''preliminary fix for the dtype issue'''
            
            if isinstance(self.loss_function, nn.CrossEntropyLoss) and model._custom_classes:
                batch_y_copy = torch.searchsorted(model.classes, batch_y_copy).long()

            self._prems_optimizer_instance.zero_grad()
            pred = model(batch_x)
            loss = self.loss_function(pred, batch_y_copy)
            loss.backward()
            self._prems_optimizer_instance.step()

            if torch.isnan(loss):
                raise ValueError('Loss is NaN')
            
    
    def _sonfis_init_optimizer(self, model, freezed_subnets):
        """
        Initializes both optimizers for use within the SONFIS training algorithm, using the premise and 
        consequent parameters of the subnets that are not frozen.

        Note:
            This method is intended exclusively for use within the SONFIS algorithm and should not be called directly in standard training.

        Args:
            model (rule_reduced_ANFIS): Rule-reduced ANFIS model to train.
            freezed_subnets (torch.Tensor): Boolean tensor indicating which subnets should not be updated.
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
        Applies the dual-optimizer training algorithm within the SONFIS training procedure, updating only the subnets that are not frozen.

        Note:
            This method is intended exclusively for use within the SONFIS algorithm and should not be called directly in standard training.

        Args:
            model (rule_reduced_ANFIS): Rule-reduced ANFIS model to train.
            train_loader (DataLoader): DataLoader containing the training data.
            val_loader (DataLoader): DataLoader containing the validation data, or ``None`` if no validation data is used.
            freezed_subnets (torch.Tensor): Boolean tensor indicating which subnets should not be updated.
        """
        self._sonfis_init_optimizer(model, freezed_subnets)
        
        # unfrozen parameters used to instantiate the optimizer to be empty (which is not allowed in PyTorch).
        if self._prems_optimizer_instance is None:
            return
        
        ep = 0
        while ep < self.epochs:
            self._update_parameters(model, train_loader)
            
            if (val_loader is not None) and (self.early_stopping is not None):
                _, val_loss = self._loss(model, train_loader, val_loader)
                self.early_stopping(model, val_loss, False)
                if self.early_stopping._stop:
                    break
                
            ep += 1
            
        if self.early_stopping is not None:
            self.early_stopping.reset()