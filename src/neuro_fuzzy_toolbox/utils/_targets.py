"""
Conversion of the targets of a model into the form expected by each computation of the toolbox.

Every conversion of targets is implemented here, so that the treatment of class labels (including custom class labels,
see ``set_custom_classes_ids``) is the same in the loss functions of the trainers, in the SONFIS operators and in the
least-squares estimation of consequents.
"""
import torch
import torch.nn as nn

# Loss functions whose targets are class indices (unless class probabilities are given)
_CLASS_INDEX_LOSSES = (nn.CrossEntropyLoss, nn.NLLLoss)


def _class_indices(model, y):
    """
    Converts class labels into class indices ``0, 1, ..., outputs - 1``.

    If the model uses custom class labels, each label is replaced by its position in ``model.classes``.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model.
        y (torch.Tensor): Class labels of shape ``(n_samples,)``.

    Returns:
        torch.Tensor: Class indices of shape ``(n_samples,)`` and dtype ``torch.int64``.
    """
    y = y.to(torch.int64)
    if model._custom_classes:
        y = torch.searchsorted(model.classes.to(y.device), y)
    return y


def _loss_targets(model, loss_function, y, pred):
    """
    Converts the targets into the form expected by a loss function, given the predictions of the model.

    For losses whose targets are class indices (``nn.CrossEntropyLoss`` and ``nn.NLLLoss``), class labels are converted
    into class indices (see :func:`_class_indices`), unless the targets already have the shape of the predictions
    (class probabilities). For any other loss, the targets are cast to the dtype of the predictions.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model.
        loss_function (torch.nn.Module): Instantiated loss function.
        y (torch.Tensor): Targets.
        pred (torch.Tensor): Predictions of the model.

    Returns:
        torch.Tensor: Targets ready to be passed to ``loss_function`` together with ``pred``.
    """
    if isinstance(loss_function, _CLASS_INDEX_LOSSES) and y.dim() == pred.dim() - 1:
        return _class_indices(model, y)
    return y if y.dtype == pred.dtype else y.to(pred.dtype)


def _compute_loss(model, loss_function, pred, y):
    """
    Computes a loss function, converting the targets as needed (see :func:`_loss_targets`).

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model.
        loss_function (torch.nn.Module): Instantiated loss function.
        pred (torch.Tensor): Predictions of the model.
        y (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Value of the loss.
    """
    return loss_function(pred, _loss_targets(model, loss_function, y, pred))


def _check_finite_loss(loss):
    """
    Raises an error if a loss value is not finite, before it is used to update the parameters.

    Args:
        loss (torch.Tensor): Value of the loss.

    Raises:
        ValueError: If the loss is ``NaN`` or infinite.
    """
    if not torch.isfinite(loss):
        raise ValueError(f'The loss is not finite ({loss.item()}). Training was stopped before updating the parameters.')


def _lse_targets(model, y, dtype):
    """
    Converts the targets of a model into the matrix used by the least-squares estimation of consequents.

    For ``output_type='softmax'``, the class labels are converted into class indices (see :func:`_class_indices`) and
    one-hot encoded. Otherwise the targets are used as they are.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model.
        y (torch.Tensor): Targets of shape ``(n_samples,)`` or ``(n_samples, outputs)``, or class labels.
        dtype (torch.dtype): Dtype of the result.

    Returns:
        torch.Tensor: Targets of shape ``(n_samples, outputs)``.
    """
    if model._output_type == 'softmax':
        y = torch.nn.functional.one_hot(_class_indices(model, y), model._outputs)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    return y.to(dtype)
