import torch
import torch.nn as nn
import torch.utils.data as data

from neuro_fuzzy_toolbox_old.training import obtain_measures

class hybrid_algorithm():
    def __init__(self, epochs, gamma=0.01, loss_function=nn.functional.mse_loss, optimizer=None, optim_params=None, validation=0, early_stopping=None):
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
        x_train = loader.dataset.tensors[0]
        y_train = loader.dataset.tensors[1].clone().detach()
        if isinstance(ANFISmodel.last_layer, nn.Softmax):
            y_train = torch.eye(ANFISmodel.consequents.shape[0])[y_train].t()
        y_train = y_train.to(x_train.dtype)

        current_consequents = ANFISmodel.consequents
        new_consequents = torch.zeros_like(ANFISmodel.consequents)

        _, w_norm, _ = ANFISmodel.intermediate_values(x_train)
        xe = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)

        if freezed == None:
            freezed = torch.zeros(ANFISmodel.rules)

        fs = w_norm.unsqueeze(2).repeat(1, 1, xe.shape[1]).view(w_norm.shape[0], -1)
        X = xe.repeat(1, ANFISmodel.rules)
        
        if current_consequents.size(0) == 1:
            flat_consequents, _, _, _ = torch.linalg.lstsq(fs * X, y_train)
            new_consequents[0] = torch.reshape(flat_consequents, (ANFISmodel.rules, xe.shape[1]))

            current_consequents[:, freezed == 0, :] = new_consequents[:, freezed == 0, :]

            ANFISmodel.set_consequents(current_consequents)

        else:
            for i in range(current_consequents.size(0)):
                flat_consequents, _, _, _ = torch.linalg.lstsq(fs * X, y_train[i])
                new_consequents[i] = torch.reshape(flat_consequents, (ANFISmodel.rules, xe.shape[1]))

            current_consequents[:, freezed == 0, :] = new_consequents[:, freezed == 0, :]

            ANFISmodel.set_consequents(current_consequents)


    def premisesUpdate(self, ANFISmodel, loader, freezed=None):
        if freezed == None:
            freezed = torch.zeros(ANFISmodel.rules)

        if (self.optimizer != None):
            optim = self.optimizer([ANFISmodel.fuzzify_layer.premises])
            if self.optim_params != None:
                self.apply_optimizer_parameters(optim)
            current_premises = ANFISmodel.premises.clone()

            for batch_x, batch_y in loader:
                batch_y_copy = batch_y.clone().detach()
                if not isinstance(ANFISmodel.last_layer, nn.Softmax):
                    batch_y_copy = batch_y_copy.to(batch_x.dtype)

                optim.zero_grad()
                pred = ANFISmodel(batch_x)
                if ANFISmodel.consequent_layer.consequents.shape[0] == 1:
                    loss = self.loss_function(pred.squeeze(), batch_y_copy)
                else:
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
                if not isinstance(ANFISmodel.last_layer, nn.Softmax):
                    batch_y_copy = batch_y_copy.to(batch_x.dtype)
                if ANFISmodel.fuzzify_layer.premises.grad != None:
                    ANFISmodel.fuzzify_layer.premises.grad.data = torch.zeros_like(ANFISmodel.fuzzify_layer.premises.grad.data)
                    ANFISmodel.consequent_layer.consequents.grad.data = torch.zeros_like(ANFISmodel.consequent_layer.consequents.grad.data)
                pred = ANFISmodel(batch_x)
                if ANFISmodel.consequent_layer.consequents.shape[0] == 1:
                    loss = self.loss_function(pred.squeeze(), batch_y_copy)
                else:
                    loss = self.loss_function(pred, batch_y_copy)
                loss.backward()

                alpha = self.gamma / torch.sqrt(torch.sum(torch.pow(ANFISmodel.fuzzify_layer.premises.grad, 2)))

                vs = new_premises[:,:,0].t()
                sigmas = new_premises[:,:,1].t()

                new_vs = torch.zeros_like(vs)
                new_sigmas = torch.zeros_like(sigmas)

                _, w_norm, outputs_by_rule = ANFISmodel.intermediate_values(batch_x)

                if isinstance(ANFISmodel.last_layer, nn.Softmax):
                    batch_y_copy = torch.eye(ANFISmodel.consequents.shape[0])[batch_y_copy]

                for k in range(ANFISmodel.rules):
                    if freezed[k] == 0:
                        A = 4*alpha*(1/torch.pow(sigmas[k], 2))*(batch_x - vs[k])
                        B = 4*alpha*(1/torch.pow(sigmas[k], 3))*torch.pow((batch_x - vs[k]), 2)
                        wk = w_norm[:,k].unsqueeze(0).t()
                        fk = outputs_by_rule[:,:,k].t()

                        if pred.ndim == 1:
                            zk = ((fk-pred)*(batch_y_copy-pred)).unsqueeze(0).t()
                        else:
                            #
                            # SUM? MEAN? PROD?
                            #
                            zk = torch.sum((fk-pred)*(batch_y_copy-pred), dim=1).unsqueeze(0).t()
                        new_vs[k] = torch.sum(A*wk*zk, dim=0)
                        new_sigmas[k] = torch.sum(B*wk*zk, dim=0)

                new_premises[:, :, 0] += new_vs.t()
                new_premises[:, :, 1] += new_sigmas.t()

            ANFISmodel.set_premises(new_premises)

    def apply_optimizer_parameters(self, optimizer):
        for param_group in optimizer.param_groups:
            for key, new_value in self.optim_params.items():
                if key in param_group:
                    param_group[key] = new_value

    def val_partition(self, loader):
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
        #Validation set
        if (val_loader != None):
            x_val = val_loader.dataset.tensors[0]
            y_val = val_loader.dataset.tensors[1]

            with torch.no_grad():
                pred = ANFISmodel(x_val)

            if isinstance(ANFISmodel.last_layer, nn.Softmax):
                val_loss = self.loss_function(pred, y_val)
            else:
                if ANFISmodel.consequent_layer.consequents.shape[0] == 1:
                    val_loss = self.loss_function(pred.squeeze(), y_val.to(pred.dtype))
                else:
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

        if isinstance(ANFISmodel.last_layer, nn.Softmax):
            loss = self.loss_function(pred, y_train)
        else:
            if ANFISmodel.consequent_layer.consequents.shape[0] == 1:
                loss = self.loss_function(pred.squeeze(), y_train.to(pred.dtype))
            else:
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