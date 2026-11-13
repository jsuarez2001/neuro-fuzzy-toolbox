"""
Conversion of the targets of a model into the form expected by each computation of the toolbox.

Every conversion of targets is implemented here, so that the treatment of class labels (including custom class labels,
see ``set_custom_classes_ids``) is the same in the loss functions of the trainers, in the SONFIS operators and in the
least-squares estimation of consequents.
"""
import math

import torch
import torch.nn as nn

# Loss functions whose targets are class indices (unless class probabilities are given)
_CLASS_INDEX_LOSSES = (nn.CrossEntropyLoss, nn.NLLLoss)

# Probability of the target class encoded by the targets of the least-squares estimation in classification (see _lse_targets)
_LSE_TARGET_PROBABILITY = 0.99


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

    The least-squares estimation fits the output of the model before its activation (the logits, in classification),
    so the targets are expressed in that same space:

    - ``output_type='default'``: the targets are used as they are.
    - ``output_type='softmax'``: each class label is converted into its class index (see :func:`_class_indices`) and
      replaced by the logits of a smoothed label distribution, which gives probability ``_LSE_TARGET_PROBABILITY``
      (:math:`p^*`) to the target class and :math:`(1 - p^*) / (K - 1)` to each of the other :math:`K - 1` classes.
      The logits are centered (they sum to zero), so the target class gets :math:`c (K - 1) / K` and every other class
      :math:`-c / K`, with :math:`c = \\log\\left(p^* (K - 1) / (1 - p^*)\\right)`. Since the softmax does not change
      when the same constant is added to every logit, the centering does not change the probabilities of the
      estimated model; it only makes the consequents of each rule sum to zero across classes.
    - ``output_type='sigmoid'``: each target :math:`q \\in [0, 1]` of each output is replaced by the logit of the
      smoothed target :math:`\\epsilon + (1 - 2 \\epsilon) q`, with :math:`\\epsilon = 1 - p^*`. Labels 0 and 1 become
      :math:`\\mp \\operatorname{logit}(p^*)`, which is the case :math:`K = 2` of the softmax rule (a single logit
      that plays the role of the difference between the two logits of the softmax).

    Encoding the class labels as one-hot vectors (or as 0/1) in the space of the logits would bound the probability of
    the target class (to :math:`e / (e + K - 1)` with a softmax) even if the targets were fitted exactly, and with a
    sigmoid it would place the decision boundary at a logit of 0.5 instead of 0.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model.
        y (torch.Tensor): Targets of shape ``(n_samples,)`` or ``(n_samples, outputs)``, or class labels.
        dtype (torch.dtype): Dtype of the result.

    Returns:
        torch.Tensor: Targets of shape ``(n_samples, outputs)``.

    Raises:
        ValueError: If ``output_type='sigmoid'`` and some target is outside ``[0, 1]``.
    """
    if model._output_type == 'softmax':
        n_classes = model._outputs
        Y = torch.nn.functional.one_hot(_class_indices(model, y), n_classes).to(dtype)
        if n_classes == 1:
            return torch.zeros_like(Y)
        p = _LSE_TARGET_PROBABILITY
        c = math.log(p * (n_classes - 1) / (1 - p))
        return c * (Y - 1 / n_classes)

    if model._output_type == 'sigmoid':
        y = y.to(dtype)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if not torch.all((y >= 0) & (y <= 1)):
            raise ValueError('With output_type=\'sigmoid\', all the targets must be between 0 and 1.')
        eps = 1 - _LSE_TARGET_PROBABILITY
        return torch.logit(eps + (1 - 2 * eps) * y)

    if y.dim() == 1:
        y = y.unsqueeze(1)
    return y.to(dtype)
