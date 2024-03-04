import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)

def obtain_measures(ANFISmodel, x, y):
    """
    Calculates various evaluation measures for the ANFIS model's predictions.

    :param ANFISmodel: The trained ANFIS model.
    :type ANFISmodel: Type3ANFIS
    
    :param x: Input data tensor.
    :type x: torch.Tensor
    
    :param y: Target output tensor.
    :type y: torch.Tensor
    
    :return: Dictionary containing different evaluation measures.
    :rtype: dict

    .. note::
        If the model its used for a regression problem, the returned measures will be the following:
            - mse: mean squared error
            - rmse: root mean squared error
            - nmse: normalized mean squared error
            - mape: mean absolute percentage error
            - mae: mean absolute error
            - R2: R2 score
            - IA: Index of Agreement
        If the model its used for a binary classification problem, the returned measures will be the following:
            - bce: binary cross entropy
            - accuracy
            - precision
            - recall
            - f1 score
    
    """
    measures = {}
    with torch.no_grad():
        pred = ANFISmodel(x)

    if isinstance(ANFISmodel.last_layer, nn.Identity):
        #pytorch calculation for IA measure
        y_mean = y.mean()
        pred_mean = pred.mean()
        IA = 1 - (torch.mean(torch.pow(y - pred, 2)) / torch.mean(torch.pow(torch.abs(y - y_mean) + torch.abs(pred - pred_mean), 2)))

        #numpy conversion for sklearn functions
        pred = pred.numpy()
        y = y.numpy()

        mse = mean_squared_error(y, pred)
        measures['mse'] = mse

        rmse = root_mean_squared_error(y, pred)
        measures['rmse'] = rmse

        variance = np.var(y)
        nmse = mse / variance
        measures['nmse'] = nmse

        #ape = np.mean(np.abs((y - pred) / y)) * 100
        #measures['ape'] = ape

        mape = mean_absolute_percentage_error(y, pred)
        measures['mape'] = mape

        mae = mean_absolute_error(y, pred)
        measures['mae'] = mae

        r2 = r2_score(y, pred)
        measures['R2'] = r2

        measures['IA'] = IA

    elif isinstance(ANFISmodel.last_layer, nn.Sigmoid):
        #pytorch calculation for bce calculation
        bce = nn.functional.binary_cross_entropy(pred, y.to(pred.dtype))
        measures['bce'] = bce

        #numpy conversion for sklearn functions
        pred = pred.numpy()
        y = y.numpy()

        threshold = 0.5
        pred = (pred >= threshold).astype(int)

        accuracy = accuracy_score(y, pred)
        measures['accuracy'] = accuracy

        precision = precision_score(y, pred)
        measures['precision'] = precision

        recall = recall_score(y, pred)
        measures['recall'] = recall

        f1 = f1_score(y, pred)
        measures['f1'] = f1

    return measures

class EarlyStopping():
    '''
    Early stopping mechanism for training algorithms of an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    **Attributes for initialization:**
    
    .. attribute:: patience
    
        Number of epochs with no improvement after which training will be stopped.
        
        :type: int
        
    .. attribute:: delta
    
        Minimum change in the monitored quantity to qualify as an improvement.
        
        :type: float
        :default: 0
        
    .. attribute:: last_state
    
        Sets whether early stopping sets the model to its latest state (otherwise sets the model to its best state).
        
        :type: bool
        :default: False
        
    - **Methods:**
        
    '''
    def __init__(self, patience, delta=0, last_state=False):
        """
        Initializes an Early Stopping mechanism to monitor the ANFIS training progress.

        :param patience: Number of epochs with no improvement after which training will be stopped.
        :type patience: int
        
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        :type delta: float
        :default delta: 0
        
        :param last_state: Sets whether early stopping sets the model to its latest state (otherwise sets the model to its best state).
        :type last_state: bool
        :default last_state: False

        """
        #Parameters
        self.patience = patience
        self.delta = delta
        self.last_state = last_state

        #For running
        self.counter = 0
        self.best_loss = None
        self.best_premises = None
        self.best_consequents = None
        self.early_stop = False

    def __call__(self, loss, ANFISmodel):
        """
        Monitors the loss during training and determines whether to stop early.

        :param loss: Current loss value.
        :type loss: float
        
        :param ANFISmodel: Instance of the ANFIS model being trained.
        :type ANFISmodel: Type3ANFIS
        
        """
        if self.best_loss is None:
            self.best_loss = loss
            self.best_premises = ANFISmodel.premises
            self.best_consequents = ANFISmodel.consequents

        elif loss + self.delta > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.last_state == False:
                    ANFISmodel.set_premises(self.best_premises)
                    ANFISmodel.set_consequents(self.best_consequents)

        else:
            self.best_loss = loss
            self.best_premises = ANFISmodel.premises
            self.best_consequents = ANFISmodel.consequents
            self.counter = 0
            
    def reset(self):
        """
        Resets the early stopping instance.

        """
        self.counter = 0
        self.best_loss = None
        self.best_premises = None
        self.best_consequents = None
        self.early_stop = False


class OLS:
    '''
    Ordinary Least Squares (OLS) optimizer for training an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    **Attributes for initialization:**
    
    .. attribute:: epochs
    
        Number of training epochs.
        
        :type: int
        
    .. attribute:: gamma
    
        Learning rate for premises update.
        
        :type: float
        :default: 0.01

    .. attribute:: loss_function
    
        Loss function for premise parameters training.
        
        :type: torch.nn.Module
        :default: nn.functional.mse_loss
        
    .. attribute:: optimizer
    
        Optimizer to use for updating premise parameters.
        
        :type: torch.optim.Optimizer
        :default: None
        
    .. attribute:: optim_params 
    
        User preference parameters for the optimizer.
        
        :type: dict
        :default: None

    .. attribute:: validation
    
        Proportion of the training data to use for validation.
        
        :type: float
        :default: 0
        
    .. attribute:: early_stopping
    
        An early stopping instance for the OLS algorithm.
        
        :type: EarlyStopping
        :default: None
        
    **Other attributes**

    .. attribute:: train_history
    
        Training loss and various measures history.
        
        :type: torch.tensor
        
    .. attribute:: val_history
    
        Validation loss and various measures history.
        
        :type: torch.tensor
    
    **Example Usage:**
    
    .. code::
    
        >>> train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), batch_size = 8)
        >>> input_data = loader.dataset.tensors[0] #x_train from loader
        >>> anfis_model = Type3ANFIS(input_data, init_rules=3)
        >>> optimizer = torch.optim.AdamW
        >>> optim_params = {'lr': 0.001, 'weight_decay': 0.001}
        >>> ols = OLS(epochs=20, optimizer=optimizer, optim_params=optim_params)
        >>> ols(anfis_model, train_loader)

    **Methods:**
    
    '''
    def __init__(self, epochs, gamma=0.01, loss_function=nn.functional.mse_loss, optimizer=None, optim_params=None, validation=0, early_stopping=None):
        """
        Initializes a new OLS instance.

        Parameters:
        - epochs (int): Number of training epochs (default: 1).
        - y (float): Learning rate for premises update (default: 0.01).
        - loss_function (torch.nn.Module): Loss function for training (default: nn.functional.mse_loss).
        - optimizer (torch.optim.Optimizer): Optimizer to use for updating parameters (default: None).
        - optim_params (dict): Additional parameters for the optimizer (default: None).
        - validation (float): Proportion of the training data to use for validation (default: 0).

        """
        #Hyperparameters
        self.epochs = epochs
        self.gamma = gamma

        #Premises Update
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.optim_params = optim_params

        #History
        self.history = {'loss': torch.tensor([])}
        self.val_history = {'loss': torch.tensor([])}

        #EarlyStopping
        self.validation = validation
        self.early_stopping = early_stopping


    def __call__(self, ANFISmodel, loader, freezed=None):
        """
        Performs training using OLS on the provided ANFIS model and data loader.

        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader
        
        :param freezed: Binary tensor indicating which rules are freezed (default: None).
        :type freezed: torch.tensor or None

        """
        train_loader, val_loader = self.val_partition(loader)

        _ = self.obtain_metrics(ANFISmodel, train_loader, val_loader)
        ep = 0
        while (ep < self.epochs):

            self.consequentsUpdate(ANFISmodel, train_loader, freezed)
            self.premisesUpdate(ANFISmodel, train_loader, freezed)

            loss = self.obtain_metrics(ANFISmodel, train_loader, val_loader)
            if self.early_stopping != None:
                self.early_stopping(loss, ANFISmodel)
                if self.early_stopping.early_stop:
                    break;

            ep += 1
        _ = self.obtain_metrics(ANFISmodel, train_loader, val_loader)


    def consequentsUpdate(self, ANFISmodel, loader, freezed=None):
        """
        Updates the consequent parameters of the ANFIS model using OLS method.
        
        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader
        
        :param freezed: Binary tensor indicating which rules are freezed (default: None).
        :type freezed: torch.tensor or None

        """
        x_train = loader.dataset.tensors[0]
        y_train = loader.dataset.tensors[1].clone().detach()
        y_train = y_train.to(x_train.dtype)

        current_consequents = ANFISmodel.consequents
        new_consequents = torch.zeros_like(ANFISmodel.consequents)

        _, w_norm, _ = ANFISmodel.intermediate_values(x_train)
        xe = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)

        if freezed == None:
            freezed = torch.zeros(ANFISmodel.rules)

        fs = w_norm.unsqueeze(2).repeat(1, 1, xe.shape[1]).view(w_norm.shape[0], -1)
        X = xe.repeat(1, ANFISmodel.rules)

        new_consequents, _, _, _ = torch.linalg.lstsq(fs * X, y_train)
        new_consequents = torch.reshape(new_consequents, (ANFISmodel.rules, xe.shape[1]))

        current_consequents[freezed == 0] = new_consequents[freezed == 0]

        ANFISmodel.set_consequents(current_consequents)


    def premisesUpdate(self, ANFISmodel, loader, freezed=None):
        """
        Updates the premise parameters of the ANFIS model using OLS method or a user preference gradient descent algorithm.
        
        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: DataLoader containing training data.
        :type loader: torch.utils.data.DataLoader
        
        :param freezed: Binary tensor indicating which rules are freezed (default: None).
        :type freezed: torch.tensor or None

        """
        if freezed == None:
            freezed = torch.zeros(ANFISmodel.consequents.shape[0])

        if (self.optimizer != None):
            optim = self.optimizer([ANFISmodel.fuzzify_layer.premises])
            if self.optim_params != None:
                self.apply_optimizer_parameters(optim)
            current_premises = ANFISmodel.premises.clone()

            for batch_x, batch_y in loader:
                batch_y_copy = batch_y.clone().detach()
                batch_y_copy = batch_y_copy.to(batch_x.dtype)

                optim.zero_grad()
                pred = ANFISmodel(batch_x)
                loss = self.loss_function(pred, batch_y_copy)
                loss.backward()
                optim.step()

            new_premises = ANFISmodel.premises.clone()
            current_premises[:,freezed==0,:] = new_premises[:,freezed==0,:]

            ANFISmodel.set_premises(current_premises)

        else:
            new_premises = ANFISmodel.premises

            for batch_x, batch_y in loader:
                batch_y_copy = batch_y.clone().detach()
                batch_y_copy = batch_y_copy.to(batch_x.dtype)
                if ANFISmodel.fuzzify_layer.premises.grad != None:
                    ANFISmodel.fuzzify_layer.premises.grad.data = torch.zeros_like(ANFISmodel.fuzzify_layer.premises.grad.data)
                    ANFISmodel.consequent_layer.consequents.grad.data = torch.zeros_like(ANFISmodel.consequent_layer.consequents.grad.data)
                pred = ANFISmodel(batch_x)
                loss = self.loss_function(pred, batch_y_copy)
                loss.backward()

                alpha = self.gamma / torch.sqrt(torch.sum(torch.pow(ANFISmodel.fuzzify_layer.premises.grad, 2)))

                vs = new_premises[:,:,0].t()
                sigmas = new_premises[:,:,1].t()

                new_vs = torch.zeros_like(vs)
                new_sigmas = torch.zeros_like(sigmas)

                _, w_norm, outputs_by_rule = ANFISmodel.intermediate_values(batch_x)

                for k in range(ANFISmodel.rules):
                    if freezed[k] == 0:
                        A = 4*alpha*(1/torch.pow(sigmas[k], 2))*(batch_x - vs[k])
                        B = 4*alpha*(1/torch.pow(sigmas[k], 3))*torch.pow((batch_x - vs[k]), 2)
                        wk = w_norm[:,k].unsqueeze(0).t()
                        fk = outputs_by_rule[:,k]
                        zk = ((fk-pred)*(batch_y-pred)).unsqueeze(0).t()

                        new_vs[k] = torch.sum(A*wk*zk, dim=0)
                        new_sigmas[k] = torch.sum(B*wk*zk, dim=0)

                new_premises[:, :, 0] += new_vs.t()
                new_premises[:, :, 1] += new_sigmas.t()

            ANFISmodel.set_premises(new_premises)
            
            
    def apply_optimizer_parameters(self, optimizer):
        """
        Apply custom optimizer parameters to the provided optimizer.

        :param optimizer: The optimizer to which parameters are applied.
        :type optimizer: torch.optim.Optimizer

        """
        for param_group in optimizer.param_groups:
            for key, new_value in self.optim_params.items():
                if key in param_group:
                    param_group[key] = new_value
                    
    
    def val_partition(self, loader):
        """
        Create separate training and validation dataloaders based on the validation split ratio.

        :param loader: The original dataloader.
        :type loader: torch.utils.data.DataLoader

        :return:
            - train_loader (torch.utils.data.DataLoader): Dataloader for training.
            - val_loader (torch.utils.data.DataLoader): Dataloader for validation.

    """
        if self.validation != 0:
            x_train, y_train = loader.dataset.tensors
            indices = torch.randperm(x_train.size(0))
            x_train = x_train[indices]
            y_train = y_train[indices]

            split_index = int(x_train.shape[0] * self.validation)

            x_val = x_train[:split_index]
            y_val = y_train[:split_index]
            x_train = x_train[split_index:]
            y_train = y_train[split_index:]

            train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), batch_size=loader.batch_size, shuffle=True)
            val_loader = data.DataLoader(data.TensorDataset(x_val, y_val), batch_size=loader.batch_size, shuffle=False)
            
        else:
            train_loader = loader
            val_loader = None

        return train_loader, val_loader
    
    def obtain_metrics(self, ANFISmodel, train_loader, val_loader):
        """
        Obtain and record various metrics such as loss and performance measures for training and validation sets.

        :param ANFISmodel: The ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param train_loader: Dataloader for training set.
        :type train_loader: torch.utils.data.DataLoader
        
        :param val_loader: Dataloader for validation set.
        :type val_loader: torch.utils.data.DataLoader

        :return loss: The calculated loss.
        :rtype loss: torch.tensor

        """
        #Validation set
        if (val_loader != None):
            x_val = val_loader.dataset.tensors[0]
            y_val = val_loader.dataset.tensors[1]

            with torch.no_grad():
                pred = ANFISmodel(x_val)

            val_loss = self.loss_function(pred, y_val.to(pred.dtype))
            self.val_history['loss'] = torch.cat([self.val_history['loss'], torch.tensor([val_loss])])

            measures = obtain_measures(ANFISmodel, x_val, y_val)

            for measure in measures:
                if measure not in self.val_history:
                    self.val_history[measure] = torch.tensor([])
                self.val_history[measure] =  torch.cat([self.val_history[measure], torch.tensor([measures[measure]])])

        #Training set
        x_train = train_loader.dataset.tensors[0]
        y_train = train_loader.dataset.tensors[1]

        with torch.no_grad():
            pred = ANFISmodel(x_train)

        loss = self.loss_function(pred, y_train.to(pred.dtype))
        self.history['loss'] = torch.cat([self.history['loss'], torch.tensor([loss])])

        measures = obtain_measures(ANFISmodel, x_train, y_train)

        for measure in measures:
            if measure not in self.history:
                self.history[measure] = torch.tensor([])
            self.history[measure] =  torch.cat([self.history[measure], torch.tensor([measures[measure]])])

        if val_loader != None:
            loss = val_loss
        return loss