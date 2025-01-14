import torch
import torch.nn as nn

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
    r2_score,
    confusion_matrix
)

def obtain_measures(ANFISmodel, x, y):
    measures = {}
    with torch.no_grad():
        pred = ANFISmodel(x)
        
    if ANFISmodel.consequent_layer.consequents.shape[0] == 1:
        pred = pred.squeeze()

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

        measures['IA'] = IA.item()

    elif isinstance(ANFISmodel.last_layer, nn.Sigmoid):
        #pytorch calculation for bce calculation
        bce = nn.functional.binary_cross_entropy(pred, y.to(pred.dtype))
        measures['bce'] = bce.item()

        #numpy conversion for sklearn functions
        pred = pred.numpy()
        y = y.numpy()
        
        threshold = 0.5
        pred = (pred >= threshold).astype(int)

        accuracy = accuracy_score(y, pred)
        measures['accuracy'] = accuracy

        precision = precision_score(y, pred, zero_division=0)
        measures['precision'] = precision

        recall = recall_score(y, pred)
        measures['recall'] = recall

        f1 = f1_score(y, pred)
        measures['f1'] = f1

        cm = confusion_matrix(y, pred)
        measures['confusion_matrix'] = cm

    elif isinstance(ANFISmodel.last_layer, nn.Softmax):
        #pytorch calculation for ce calculation
        ce = nn.functional.cross_entropy(pred, y)
        measures['ce'] = ce.item()

        pred = ANFISmodel.classes_prediction(x)

        #numpy conversion for sklearn functions
        pred = pred.numpy()
        y = y.numpy()

        accuracy = accuracy_score(y, pred)
        measures['accuracy'] = accuracy

        precision = precision_score(y, pred, average='weighted', zero_division=0)
        measures['precision'] = precision

        recall = recall_score(y, pred, average='weighted')
        measures['recall'] = recall

        f1 = f1_score(y, pred, average='weighted')
        measures['f1'] = f1

        cm = confusion_matrix(y, pred, labels=list(range(ANFISmodel.consequents.shape[0])))
        measures['confusion_matrix'] = cm

    return measures