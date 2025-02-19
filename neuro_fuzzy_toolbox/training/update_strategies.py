import torch

def classical_consequents_estimation_with_OLS(ANFISmodel, loader):
    """
    Estima los parámetros consecuentes de un modelo ANFIS utilizando mínimos cuadrados ordinarios.
    
    Args:
        ANFISmodel (ANFIS | h_ANFIS): Modelo ANFIS a entrenar.
        loader (DataLoader): DataLoader con los datos de entrenamiento.
        
    Returns:
        torch.tensor: Tensor con los nuevos parámetros consecuentes.
    """
    x = loader.dataset.tensors[0]
    y = loader.dataset.tensors[1]
        
    # Least squares problem construction
    _, w_norm, _ = ANFISmodel.intermediate_values(x)
    xe = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    fs = w_norm.unsqueeze(2).repeat(1, 1, xe.shape[1]).view(w_norm.shape[0], -1)
    X = xe.repeat(1, ANFISmodel.rules)
        
    '''preliminary fix for the dtype issue'''
    if ANFISmodel._output_type == 'multiclass':
        y = torch.nn.functional.one_hot(y, ANFISmodel._outputs)
    if y.dtype != X.dtype:
        y = y.to(X.dtype)
    '''preliminary fix for the dtype issue'''
    
    # Solve least squares problem using QR decomposition with pivoting
    C, _, _, _ = torch.linalg.lstsq(X * fs, y)
    new_consequents = C.t().reshape(ANFISmodel._outputs, ANFISmodel.rules, xe.shape[1])
    
    # Update consequents
    return new_consequents


def optimizer_training_epoch(model, loader, optimizer, loss_function):
    """
    Actualiza los parámetros de un modelo utilizando un optimizador y una función de pérdida dados como parámetros. Los parámetros a actualizar son indicados por el optimizador (ya instanciado).
    
    Args:
        model (nn.Module): Modelo ANFIS a entrenar.
        loader (DataLoader): DataLoader con los datos de entrenamiento.
        optimizer (torch.optim.Optimizer): Optimizador instanciado a utilizar.
        loss_function (callable): Función de pérdida a utilizar.
    
    """
    for batch_x, batch_y in loader:
        batch_y_copy = batch_y.clone().detach()
        '''preliminary fix for the dtype issue'''
        if loader.dataset.tensors[0].dtype != loader.dataset.tensors[1].dtype:
            batch_y_copy = batch_y_copy.to(batch_x.dtype)
        '''preliminary fix for the dtype issue'''
        
        '''cross_entropy function only accepts torch.long (torch.int64) dtype for target indices'''
        if loss_function == torch.nn.functional.cross_entropy:
            batch_y_copy = batch_y_copy.type(torch.int64)
        '''cross_entropy function only accepts torch.long (torch.int64) dtype for target indices'''
        
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_function(pred, batch_y_copy)
        loss.backward()
        optimizer.step()
        
        if torch.isnan(loss):
            print("--- prem grads --- prem grads --- prem grads ----")
            for i in range (model._input_size):
                print(model._fuzzification_layer._premises[i].grad)
                
            print("")
            print("--- prem param --- prem param --- prem param ----")
            print(model.premises_structure)
            print("")
            
            raise ValueError('Loss is NaN')
        