import torch
import torch.nn as nn


class OLS:
    '''
    Ordinary Least Squares (OLS) optimizer for training an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    **Attributes:**
    
    .. attribute:: epochs
    
        Number of training epochs.
        
        :type: int
        
    .. attribute:: y
    
        Learning rate for premises update.
        
        :type: float
        :default: 0.01

    .. attribute:: loss_function
    
        Loss function for training.
        
        :type: torch.nn.Module
        :default: nn.functional.mse_loss

    .. attribute:: validation
    
        Proportion of the training data to use for validation.
        
        :type: float
        :default: 0

    .. attribute:: train_history
    
        Training loss history.
        
        :type: torch.tensor
        
    .. attribute:: val_history
    
        Validation loss history.
        
        :type: torch.tensor
        
    **To initialize it:**
        
    The parameters that must be taken into account are the following:

    :param epochs: Number of training epochs.
    :type epochs: int

    :param y: Learning rate for premises update (default: 0.01).
    :type y: float
        
    :param loss_function: Loss function for training (default: nn.functional.mse_loss).
    :type loss_function: torch.nn.Module
        
    :param validation: Proportion of the training data to use for validation (default: 0).
    :type validation: float
    
    **Example Usage:**
    
    .. code::
    
        >>> input_data = torch.rand(100,2)
        >>> anfis_model = Type3ANFIS(input_data, init_rules=3)
        >>> train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), batch_size = 8)
        >>> ols_optimizer = OLS(20, y=0.001)
        >>> ols_optimizer(anfis_model, train_loader)

    **Methods:**
    
    '''
    def __init__(self, epochs, y=0.01, loss_function=nn.functional.mse_loss, validation=0):
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

        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param loader: Data loader for training.
        :type loader: torch.utils.data.DataLoader
        
        :param freezed: Binary tensor indicating which premises to freeze (default: None).
        :type freezed: torch.tensor or None

        """
        x_train = loader.dataset.tensors[0]
        y_train = loader.dataset.tensors[1]

        if (self.validation != 0):
            split_index = int(x_train.shape[0] * self.validation)
            x_train, x_val = torch.split(x_train, [split_index, x_train.shape[0] - split_index])
            y_train, y_val = torch.split(y_train, [split_index, y_train.shape[0] - split_index])

        ANFISmodel.set_consequents(self.consequentsUpdate(ANFISmodel, x_train, y_train, freezed))
        ep = 0
        while (ep < self.epochs):
            ANFISmodel.set_premises(self.premisesUpdate(ANFISmodel, x_train, y_train, self.y, self.loss_function, freezed))
            ep += 1
        ANFISmodel.set_consequents(self.consequentsUpdate(ANFISmodel, x_train, y_train, freezed))

        with torch.no_grad():
            if (self.validation != 0):
                val_loss = self.loss_function(ANFISmodel(x_val), y_val)
                self.val_history = torch.cat([self.val_history, torch.tensor([val_loss])])

            loss = self.loss_function(ANFISmodel(x_train), y_train)
            self.train_history = torch.cat([self.train_history, torch.tensor([loss])])

    def consequentsUpdate(self, ANFISmodel, x_train, y_train, freezed=None):
        """
        Updates the consequents (linear coefficients) of the ANFIS model.
        
        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param x_train: Input training data.
        :type x_train: torch.tensor
        
        :param y_train: Target training data.
        :type y_train: torch.tensor
        
        :param freezed: Binary tensor indicating which premises to freeze (default: None).
        :type freezed: torch.tensor or None
        
        :return: Updated consequent parameters.
        :rtype: torch.tensor

        """
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

        return new_consequents;

    def premisesUpdate(self, ANFISmodel, x_train, y_train, y=0.01, loss_function=nn.functional.mse_loss, freezed=None):
        """
        Updates the premises (fuzzy sets) of the ANFIS model.
        
        :param ANFISmodel: An instance of the Type3ANFIS model.
        :type ANFISmodel: Type3ANFIS
        
        :param x_train: Input training data.
        :type x_train: torch.tensor
        
        :param y_train: Target training data.
        :type y_train: torch.tensor
        
        :param y: Learning rate for premises update (default: 0.01).
        :type y: float
        
        :param loss_function: Loss function for training (default: nn.functional.mse_loss).
        :type loss_function: torch.nn.Module
        
        :param freezed: Binary tensor indicating which premises to freeze (default: None).
        :type freezed: torch.tensor or None
        
        :return: Updated fuzzy premises.
        :rtype: torch.tensor

        """
        pred = ANFISmodel(x_train)

        loss = loss_function(pred, y_train)
        loss.backward()

        alpha = y / torch.sqrt(torch.sum(torch.pow(ANFISmodel.fuzzify_layer.premises.grad, 2)))

        if freezed == None:
            freezed = torch.zeros(ANFISmodel.consequents.shape[0])

        new_premises = ANFISmodel.premises

        vs = new_premises[:,:,0].t()
        sigmas = new_premises[:,:,1].t()

        new_vs = torch.zeros_like(vs)
        new_sigmas = torch.zeros_like(sigmas)

        _, w_norm, outputs_by_rule = ANFISmodel.intermediate_values(x_train)

        for k in range(ANFISmodel.rules):
            if freezed[k] == 0:
                A = 4*alpha*(1/torch.pow(sigmas[k], 2))*(x_train - vs[k])
                B = 4*alpha*(1/torch.pow(sigmas[k], 3))*torch.pow((x_train - vs[k]), 2)
                wk = w_norm[:,k].unsqueeze(0).t()
                fk = outputs_by_rule[:,k]
                zk = ((fk-pred)*(y_train-pred)).unsqueeze(0).t()

                new_vs[k] = torch.sum(A*wk*zk, dim=0)
                new_sigmas[k] = torch.sum(B*wk*zk, dim=0)

        new_premises[:, :, 0] += new_vs.t()
        new_premises[:, :, 1] += new_sigmas.t()

        return new_premises;



