import torch
import torch.nn as nn

def classical_consequents_estimation_with_OLS(ANFISmodel, loader, driver, ridge_lambda):
    """
    Estimates the consequent parameters of an ANFIS model using ordinary least squares.

    Note:
        Specifically, QR decomposition with pivoting is used to solve the least-squares
        problem. For more information, see: https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html.

    Args:
        ANFISmodel (ANFIS | h_ANFIS): ANFIS model whose consequent parameters are to be estimated.
        loader (DataLoader): DataLoader containing the training data.
        driver (str): Backend function to use for the least-squares estimation. 
            Valid values are ``'gels'``, ``'gelsy'``, ``'gelsd'``, and ``'gelss'``. If ``None``, defaults to ``'gels'``.
        ridge_lambda (float): Lambda value for Ridge regularization in the least-squares estimation.
            If ``0.``, no regularization is applied.

    Returns:
        torch.Tensor: Tensor containing the new consequent parameters.
    """
    x = loader.dataset.tensors[0]
    y = loader.dataset.tensors[1]
    
    # Least squares problem construction
    w_norm = ANFISmodel.get_firing_levels(x, normalized=True)
    xe = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    fs = w_norm.unsqueeze(2).repeat(1, 1, xe.shape[1]).view(w_norm.shape[0], -1)
    X = xe.repeat(1, ANFISmodel.rules)
        
    '''preliminary fix for the dtype issue'''
    if ANFISmodel._output_type == 'softmax':
        y = y.to(torch.int64)
        y = torch.nn.functional.one_hot(y, ANFISmodel._outputs)
    if y.dtype != X.dtype:
        y = y.to(X.dtype)
    '''preliminary fix for the dtype issue'''
    
    A = X * fs
    
    if ridge_lambda > 0.:
        p = A.shape[1]
        I = torch.eye(p, dtype=A.dtype) * torch.sqrt(torch.tensor(ridge_lambda, dtype=A.dtype))
        A = torch.cat([A, I], dim=0)
        if y.dim() > 1:
            m = y.shape[1]
            zeros = torch.zeros((p, m), dtype=A.dtype)
        else:
            zeros = torch.zeros(p, dtype=A.dtype)
        y  = torch.cat([y, zeros], dim=0)
    
    # Solve least squares problem using QR decomposition with pivoting
    C, _, _, _ = torch.linalg.lstsq(A, y, rcond=None, driver=driver)
    new_consequents = C.t().reshape(ANFISmodel._outputs, ANFISmodel.rules, xe.shape[1])
    
    return new_consequents


def optimizer_training_epoch(model, loader, optimizer, loss_function):
    """
    Updates the parameters of a model for one training epoch using a given optimizer and loss function. 
    The parameters to be updated are determined by the optimizer.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
        loader (DataLoader): DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): Instantiated optimizer to use.
        loss_function (torch.nn.Module): Loss function to use.
    """
    for batch_x, batch_y in loader:
        batch_y_copy = batch_y.clone().detach()
        
        '''preliminary fix for the dtype issue'''
        if not isinstance(loss_function, nn.CrossEntropyLoss): #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            if loader.dataset.tensors[0].dtype != loader.dataset.tensors[1].dtype:
                batch_y_copy = batch_y_copy.to(batch_x.dtype)
        else:
            batch_y_copy = batch_y_copy.long()
        '''preliminary fix for the dtype issue'''
        
        if isinstance(loss_function, nn.CrossEntropyLoss) and model._custom_classes:
            batch_y_copy = torch.searchsorted(model.classes, batch_y_copy).long()
        
        optimizer.zero_grad()
        pred = model(batch_x)
        
        loss = loss_function(pred, batch_y_copy)
        loss.backward()
        
        optimizer.step()
        
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')