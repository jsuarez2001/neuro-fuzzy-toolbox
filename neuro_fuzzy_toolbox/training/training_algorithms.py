import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split

from neuro_fuzzy_toolbox.training import (
    classical_consequents_estimation_with_OLS,
    premises_update_with_gradient_descent
)

class Hybrid_learning_algorithm():
    def __init__(self, epochs, loss_function, validation=0, early_stopping=None, optimizer=torch.optim.Adam, optimizer_params={}):
        # Training parameters
        self.epochs = epochs
        self.loss_function = loss_function
        
        # Optimizer
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        
        # Early stopping
        self.validation = validation
        self.early_stopping = early_stopping
        
        # History
        self.history = {"loss": []}
        self.val_history = {"loss": []}
        
        
    def __call__(self, ANFISmodel, loader, verbose=True):
        train_loader, val_loader = self._train_val_split(loader)
        
        self._register_loss(ANFISmodel, train_loader, val_loader)

        ep = 0
        while ep < self.epochs:
            
            ANFISmodel.set_consequents(self._consequents_update(ANFISmodel, train_loader))
            self._premises_update(ANFISmodel, train_loader)
            
            self._register_loss(ANFISmodel, train_loader, val_loader)
            
            loss, val_loss = self.history["loss"][-1], self.val_history["loss"][-1]
            
            if self.validation > 0 and self._check_early_stop(ANFISmodel, val_loss, verbose):
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
            
        
    def _consequents_update(self, ANFISmodel, loader):
        return classical_consequents_estimation_with_OLS(ANFISmodel, loader)
    
    
    def _premises_update(self, ANFISmodel, loader):
        # Optimizer
        optimizer = self.optimizer([ANFISmodel._fuzzification_layer._premises])
        self._apply_optimizer_parameters(optimizer)
        
        premises_update_with_gradient_descent(ANFISmodel, loader, optimizer, self.loss_function)
    
    def _loss_function(self, pred, y):
        '''preliminary fix for the dtype issue'''
        if y.dtype != pred.dtype:
            y = y.to(pred.dtype)
        '''preliminary fix for the dtype issue'''
        
        '''torch cross_entropy function only accepts torch.long (torch.int64) dtype for target indices'''
        if self.loss_function == torch.nn.functional.cross_entropy:
            y = y.type(torch.int64)
        '''torch cross_entropy function only accepts torch.long (torch.int64) dtype for target indices'''
            
        return self.loss_function(pred, y)
    
    def _apply_optimizer_parameters(self, optimizer):
        if self.optimizer_params != {}:
            for param_group in optimizer.param_groups:
                for key, new_value in self.optimizer_params.items():
                    if key in param_group:
                        param_group[key] = new_value
                        
    
    def _train_val_split(self, train_loader):
        val_loader = None
        
        if self.validation != 0:
            x_train, y_train = train_loader.dataset.tensors
            x_train, x_val, y_train, y_val = train_test_split(x_train.numpy(), y_train.numpy(), test_size=self.validation, shuffle=True)
            x_train, x_val, y_train, y_val = torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(y_train), torch.from_numpy(y_val)
            
            train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), batch_size=train_loader.batch_size, shuffle=True)
            val_loader = data.DataLoader(data.TensorDataset(x_val, y_val), batch_size=train_loader.batch_size, shuffle=False)

        return train_loader, val_loader
    
    
    def _check_early_stop(self, ANFISmodel, loss, verbose):
        if self.early_stopping is not None:
            self.early_stopping(ANFISmodel, loss, verbose)
            if self.early_stopping._stop:
                return True
        return False
    
    
    def _loss(self, ANFISmodel, train_loader, val_loader):
        val_loss = None
        x = train_loader.dataset.tensors[0]
        y = train_loader.dataset.tensors[1]
        with torch.no_grad():
            pred = ANFISmodel(x)
            loss = self._loss_function(pred, y)
        if self.validation > 0:
            x = val_loader.dataset.tensors[0]
            y = val_loader.dataset.tensors[1]
            with torch.no_grad():
                pred = ANFISmodel(x)
                val_loss = self._loss_function(pred, y)
        return loss, val_loss
    
    
    def _register_loss(self, ANFISmodel, train_loader, val_loader):
        loss, val_loss = self._loss(ANFISmodel, train_loader, val_loader)
        self.history["loss"].append(loss.item())
        if self.validation > 0:
            self.val_history["loss"].append(val_loss.item())
