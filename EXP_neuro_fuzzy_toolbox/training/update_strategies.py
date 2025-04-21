import torch

def classical_consequents_estimation_with_OLS(ANFISmodel, loader):
    """
    Estima los parámetros consecuentes de un modelo ANFIS utilizando mínimos cuadrados ordinarios.
    
    Note:
        Específicamente, se usa la descomposición QR con pivoteo para resolver el problema de mínimos cuadrados. Para más información, ver: https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html.
      
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
    if ANFISmodel._output_type == 'softmax':
        y = y.to(torch.int64)
        y = torch.nn.functional.one_hot(y, ANFISmodel._outputs)
    if y.dtype != X.dtype:
        y = y.to(X.dtype)
    '''preliminary fix for the dtype issue'''
    
    # Solve least squares problem using QR decomposition with pivoting
    C, _, _, _ = torch.linalg.lstsq(X * fs, y, rcond=None, driver=None)
    new_consequents = C.t().reshape(ANFISmodel._outputs, ANFISmodel.rules, xe.shape[1])
    
    return new_consequents


def optimizer_training_epoch(model, loader, optimizer, loss_function):
    """
    Actualiza los parámetros de un modelo utilizando un optimizador (ya instanciado) y una función de pérdida dados como parámetros. Los parámetros a actualizar son indicados por el optimizador (ya instanciado).
    
    Args:
        model (nn.Module): Modelo a entrenar.
        loader (DataLoader): DataLoader con los datos de entrenamiento.
        optimizer (torch.optim.Optimizer): Optimizador instanciado a utilizar.
        loss_function (callable): Función de pérdida a utilizar.
    
    """
    #print("-----------------------------")
    #print("premises grads")
    #print(model._fuzzification_layer._premises.grad)
    #print("premises")
    #print(model.get_premises())
    for batch_x, batch_y in loader:
        batch_y_copy = batch_y.clone().detach()
        
        '''preliminary fix for the dtype issue'''
        if loss_function != torch.nn.functional.cross_entropy: #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            if loader.dataset.tensors[0].dtype != loader.dataset.tensors[1].dtype:
                batch_y_copy = batch_y_copy.to(batch_x.dtype)
        else: 
            batch_y_copy = batch_y_copy.to(torch.int64)
        '''preliminary fix for the dtype issue'''
        
        optimizer.zero_grad()
        pred = model(batch_x)
        
        #print("pred")
        #print(pred.max())
        #print(pred.min())
        #print("")
        #print("batch_y_copy")
        #print(batch_y_copy.max())
        #print(batch_y_copy.min())
        #print("")
        
        loss = loss_function(pred, batch_y_copy)
        loss.backward()
        
        #print("-----------------------------")
        #print("premises grads")
        #print(model._fuzzification_layer._premises.grad)
        
        optimizer.step()
        
        #print("premises")
        #print(model.get_premises())
        
        if torch.isnan(loss):
            print("--- prem grads --- prem grads --- prem grads ----")
            for i in range (model._input_size):
                print(model._fuzzification_layer._premises[i].grad)
                
            print("")
            print("--- prem param --- prem param --- prem param ----")
            print(model.premises_structure)
            print("")
            
            raise ValueError('Loss is NaN')
        