import torch
import torch.nn as nn

def classical_consequents_estimation_with_OLS(ANFISmodel, loader, driver, ridge_lambda):
    """
    Estima los parámetros consecuentes de un modelo ANFIS utilizando mínimos cuadrados ordinarios.
    
    Note:
        Específicamente, se usa la descomposición QR con pivoteo para resolver el problema de mínimos cuadrados. Para más información, ver: https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html.
      
    Args:
        ANFISmodel (ANFIS | h_ANFIS): Modelo ANFIS a entrenar.
        loader (DataLoader): DataLoader con los datos de entrenamiento.
        driver (string): Chooses the backend function that will be used, vaild values are: 'gels', 'gelsy', 'gelsd', 'gelss'. If None, then uses 'gels' (Default: None)
        ridge_lambda (float): Lambda usado para utilizar "Regularización Ridge" en la estimación de consecuentes con mínimos cuadrados. Si es 0, no se realiza regularización.
        
    Returns:
        torch.tensor: Tensor con los nuevos parámetros consecuentes.
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
    Actualiza los parámetros de un modelo utilizando un optimizador (ya instanciado) y una función de pérdida dados como parámetros. Los parámetros a actualizar son indicados por el optimizador (ya instanciado).
    
    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): Modelo ANFIS a entrenar.
        loader (DataLoader): DataLoader con los datos de entrenamiento.
        optimizer (torch.optim.Optimizer): Optimizador instanciado a utilizar.
        loss_function (torch.nn.Module): Función de pérdida a utilizar.
    
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