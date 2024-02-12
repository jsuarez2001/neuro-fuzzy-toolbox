import torch
import torch.nn as nn


class OLS:
    '''
    Ordinary Least Squares (OLS) optimizer for training an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    Attributes:
    - epochs (int): Number of training epochs (default: 1).
    - y (float): Learning rate for premises update (default: 0.01).
    - loss_function (torch.nn.Module): Loss function for training (default: nn.functional.mse_loss).
    - validation (float): Proportion of the training data to use for validation (default: 0).
    - train_history (torch.Tensor): Training loss history.
    - val_history (torch.Tensor): Validation loss history.

    Methods:
    - __init__: Initializes a new OLS instance.
    - __call__: Performs training using OLS on the provided ANFIS model and data loader.
    - consequentsUpdate: Updates the consequents (linear coefficients) of the ANFIS model.
    - premisesUpdate: Updates the premises (fuzzy sets) of the ANFIS model.

    Example Usage:
    >>> ols_optimizer = OLS(y=0.001)
    >>> anfis_model = Type3ANFIS(input_size=4, init_rules=3)
    >>> train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), batch_size = 8)
    >>> ols_optimizer(anfis_model, train_loader)

    '''
    def __init__(self, epochs=1, y=0.01, loss_function=nn.functional.mse_loss, validation=0):
        """
        Initializes a new OLS instance.

        Parameters:
        - epochs (int): Number of training epochs (default: 1).
        - y (float): Learning rate for premises update (default: 0.01).
        - loss_function (torch.nn.Module): Loss function for training (default: nn.functional.mse_loss).
        - validation (float): Proportion of the training data to use for validation (default: 0).

        """
        self.epochs = epochs
        self.y = y
        self.loss_function = loss_function
        self.validation = validation
        self.train_history = torch.tensor([])
        self.val_history = torch.tensor([])

    def __call__(self, ANFISmodel, loader, freezed=None):
        """
        Performs training using OLS on the provided ANFIS model and data loader.

        Parameters:
        - ANFISmodel (Type3ANFIS): An instance of the Type3ANFIS model.
        - loader (torch.utils.data.DataLoader): Data loader for training.
        - freezed (torch.Tensor or None): Binary tensor indicating which premises to freeze (default: None).

        """
        ep = 0
        while (ep < self.epochs):
            x_train = loader.dataset.tensors[0]
            y_train = loader.dataset.tensors[1]

            if (self.validation != 0):
                split_index = int(x_train.shape[0] * self.validation)
                x_train, x_val = torch.split(x_train, [split_index, x_train.shape[0] - split_index])
                y_train, y_val = torch.split(y_train, [split_index, y_train.shape[0] - split_index])

            ANFISmodel.set_consequents(self.consequentsUpdate(ANFISmodel, x_train, y_train, freezed))
            ANFISmodel.set_premises(self.premisesUpdate(ANFISmodel, x_train, y_train, self.y, self.loss_function, freezed))

            with torch.no_grad():
                if (self.validation != 0):
                    val_loss = self.loss_function(ANFISmodel(x_val), y_val)
                    self.val_history = torch.cat([self.val_history, torch.tensor([val_loss])])

                loss = self.loss_function(ANFISmodel(x_train), y_train)
                self.train_history = torch.cat([self.train_history, torch.tensor([loss])])
            
            ep += 1

    def consequentsUpdate(self, ANFISmodel, x_train, y_train, freezed=None):
        """
        Updates the consequents (linear coefficients) of the ANFIS model.

        Parameters:
        - ANFISmodel (Type3ANFIS): An instance of the Type3ANFIS model.
        - x_train (torch.Tensor): Input training data.
        - y_train (torch.Tensor): Output training data.
        - freezed (torch.Tensor or None): Binary tensor indicating which consequents to freeze (default: None).

        Returns:
        - new_consequents (torch.Tensor): Updated linear coefficients.

        """
        new_consequents = torch.zeros_like(ANFISmodel.consequents)
        
        # Calculate normalized firing levels using the ANFIS model
        w_norm, _ = ANFISmodel.intermediate_values(x_train)
        
        # Extend input data with a column of ones
        xe = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)
        
        # Check if any consequents are frozen
        if freezed == None:
            freezed = torch.zeros(new_consequents.shape[0])
        
        # Update each consequent parameters by rule
        for i in range(new_consequents.shape[0]):
            if freezed[i] == 0:
                # weight matrix is a diagonal matrix with the firing levels of each data dimension by rule
                w_diag = torch.diag(w_norm[:, i])

                # Perform least squares solution to update the consequents
                new_consequents[i], _, _, _ = torch.linalg.lstsq(w_diag @ xe, w_diag @ y_train)

        return new_consequents;

    def premisesUpdate(self, ANFISmodel, x_train, y_train, y=0.01, loss_function=nn.functional.mse_loss, freezed=None):
        """
        Updates the premises (fuzzy sets) of the ANFIS model.

        Parameters:
        - ANFISmodel (Type3ANFIS): An instance of the Type3ANFIS model.
        - x_train (torch.Tensor): Input training data.
        - y_train (torch.Tensor): Output training data.
        - y (float): Learning rate for premises update (default: 0.01).
        - loss_function (torch.nn.Module): Loss function for training (default: nn.functional.mse_loss).
        - freezed (torch.Tensor or None): Binary tensor indicating which premises to freeze (default: None).

        Returns:
        - new_premises (torch.Tensor): Updated fuzzy premises.

        """
        # Forward pass to obtain model predictions
        pred = ANFISmodel(x_train)

        # Calculate loss and perform backward pass
        loss = loss_function(pred, y_train)
        loss.backward()

        # Calculate the step size alpha for the update
        alpha = y / torch.sqrt(torch.sum(torch.pow(ANFISmodel.fuzzify_layer.premises.grad, 2)))
        
        # Check if any rules are frozen
        if freezed == None:
            freezed = torch.zeros(ANFISmodel.consequents.shape[0])
            
        # Initialize new premises with the current premises
        new_premises = ANFISmodel.premises
        
        # Extract parameters from the premises (mu and sigma)
        vs = new_premises[:,:,0].t()
        sigmas = new_premises[:,:,1].t()
        
        # Initialize tensors for the updated premises
        new_vs = torch.zeros_like(vs)
        new_sigmas = torch.zeros_like(sigmas)
        
        # Get normalized firing levels and outputs by rule
        w_norm, outputs_by_rule = ANFISmodel.intermediate_values(x_train)
        
        # Update premises by rule
        for k in range(ANFISmodel.rules):
            if freezed[k] == 0:
                A = 4 * alpha * (1 / torch.pow(sigmas[k], 2)) * (x_train - vs[k])
                B = 4 * alpha * (1 / torch.pow(sigmas[k], 3)) * torch.pow((x_train - vs[k]), 2)
                wk = w_norm[:, k].unsqueeze(0).t()
                fk = outputs_by_rule[:, k]
                zk = ((fk - pred) * (y_train - pred)).unsqueeze(0).t()
                
                # Update mu (vs) and sigma (sigmas) for each rule
                new_vs[k] = torch.sum(A * wk * zk, dim=0)
                new_sigmas[k] = torch.sum(B * wk * zk, dim=0)
                
        # Update the premises with the new values
        new_premises[:, :, 0] += new_vs.t()
        new_premises[:, :, 1] += new_sigmas.t()

        return new_premises
