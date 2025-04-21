from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
    accuracy_score,
    r2_score
)

def get_measures(ANFISmodel, x, y):
    """
    Genera un diccionario con distintas medidas de evaluación de un modelo ANFIS dependiendo de su tipo de salida.

    Args:
        ANFISmodel (ANFIS | h_ANFIS): Modelo ANFIS a evaluar.
        x (torch.Tensor): Datos de entrada.
        y (torch.Tensor): Datos de salida.

    Returns:
        dict: Diccionario con las métricas de evaluación del modelo.
    """
    measures = {}
    y = y.numpy()
    pred = ANFISmodel.predict(x)

    if ANFISmodel._output_type == 'default':
        measures['MSE'] = mean_squared_error(y, pred)
        measures['RMSE'] = root_mean_squared_error(y, pred)
        measures['MAE'] = mean_absolute_error(y, pred)
        measures['R2'] = r2_score(y, pred)
        measures['MAPE'] = mean_absolute_percentage_error(y, pred)

    elif ANFISmodel._output_type == 'sigmoid':
        measures['Accuracy'] = accuracy_score(y, pred)
        measures['Precision'] = precision_score(y, pred, zero_division=0)
        measures['Recall'] = recall_score(y, pred)
        measures['F1'] = f1_score(y, pred, zero_division=0)
        measures['Confusion Matrix'] = confusion_matrix(y, pred)

    elif ANFISmodel._output_type == 'softmax':
        measures['Accuracy'] = accuracy_score(y, pred)
        measures['Precision'] = precision_score(y, pred, average='weighted', zero_division=0)
        measures['Recall'] = recall_score(y, pred, average='weighted', zero_division=0)
        measures['F1'] = f1_score(y, pred, average='weighted', zero_division=0)
        measures['Confusion Matrix'] = confusion_matrix(y, pred, labels=list(range(ANFISmodel._outputs)))

    return measures