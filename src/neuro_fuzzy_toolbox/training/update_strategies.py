import torch

from neuro_fuzzy_toolbox.utils._loader import get_loader_tensors
from neuro_fuzzy_toolbox.utils._least_squares import _global_consequents_estimation, _local_consequents_estimation
from neuro_fuzzy_toolbox.utils._targets import _compute_loss, _check_finite_loss

def classical_consequents_estimation_with_OLS(ANFISmodel, loader, driver, ridge_lambda, freezed_subnets=None):
    """
    Estimates the consequent parameters of an ANFIS model using ordinary least squares.
    
    If ``freezed_subnets`` is provided, only the consequent parameters of the rules that are not frozen are
    estimated, and the frozen rules keep their current consequents. In that case, the least-squares problem is
    solved on the residual left by the frozen rules (the targets minus the contribution of the frozen rules to
    the model output), which yields the consequents of the active rules that minimize the squared error of the
    whole model given the frozen ones.

    Note:
        Specifically, QR decomposition with pivoting is used to solve the least-squares
        problem. For more information, see: https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html.

    Args:
        ANFISmodel (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model whose consequent parameters are to be estimated.
        loader (DataLoader): DataLoader containing the training data.
        driver (str): Backend function to use for the least-squares estimation. 
            Valid values are ``'gels'``, ``'gelsy'``, ``'gelsd'``, and ``'gelss'``. If ``None``, defaults to ``'gels'``.
        ridge_lambda (float): Lambda value for Ridge regularization in the least-squares estimation.
            If ``0.``, no regularization is applied.
        freezed_subnets (torch.Tensor, optional): Boolean tensor of length ``num_rules`` indicating which rules keep
            their current consequent parameters. If ``None``, the consequents of all the rules are estimated.
            Defaults to ``None``.

    Returns:
        torch.Tensor: Tensor containing the new consequent parameters, of shape ``(outputs, rules, input_size + 1)``.
    """
    x, y = get_loader_tensors(loader)
    return _global_consequents_estimation(ANFISmodel, x, y, driver, ridge_lambda, freezed_subnets)


def local_consequents_estimation_with_WLS(ANFISmodel, loader, driver, ridge_lambda, freezed_subnets=None):
    """
    Estimates the consequent parameters of each rule separately, by weighted least squares (local estimation).
    
    For each rule :math:`k`, the consequent parameters :math:`\\Theta_k` solve
    
    .. math::
    
        \\min_{\\Theta_k} \\sum_{t=1}^{N} w_k(\\mathbf{x}_t) \\left( \\mathbf{y}_t - \\mathbf{x}_{e,t}^\\top \\Theta_k \\right)^2,
    
    where :math:`w_k` is the (unnormalized) firing level of the rule and :math:`\\mathbf{x}_{e,t} = [\\mathbf{x}_t; 1]`.
    Each rule thus fits a local linear model of the target over the region of the input space it covers, independently
    of the other rules. This is the estimation used by the hybrid learning procedure of the original SONFIS formulation
    (Allende-Cid et al., 2016). In contrast with :func:`classical_consequents_estimation_with_OLS`, it does not minimize
    the error of the whole model, but each consequent can be read on its own as a local model, and its cost grows
    linearly (instead of cubically) with the number of rules.

    Note:
        The firing levels of each rule are divided by their maximum before solving, which does not change the solution 
        but makes the Ridge penalty comparable across rules. Rules that are not activated by any sample keep their current
        consequent parameters.

    Args:
        ANFISmodel (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model whose consequent parameters are to be estimated.
        loader (DataLoader): DataLoader containing the training data.
        driver (str): Backend function to use for the least-squares estimation. 
            Valid values are ``'gels'``, ``'gelsy'``, ``'gelsd'``, and ``'gelss'``. If ``None``, the default of
            :func:`torch.linalg.lstsq` is used.
        ridge_lambda (float): Lambda value for Ridge regularization in the least-squares estimation.
            If ``0.``, no regularization is applied.
        freezed_subnets (torch.Tensor, optional): Boolean tensor of length ``num_rules`` indicating which rules keep
            their current consequent parameters. Since the rules are estimated independently, the frozen rules are
            simply skipped. Defaults to ``None``.

    Returns:
        torch.Tensor: Tensor containing the new consequent parameters, of shape ``(outputs, rules, input_size + 1)``.
    """
    x, y = get_loader_tensors(loader)
    return _local_consequents_estimation(ANFISmodel, x, y, driver, ridge_lambda, freezed_subnets)


def optimizer_training_epoch(model, loader, optimizer, loss_function):
    """
    Updates the parameters of a model for one training epoch using a given optimizer and loss function. 
    The parameters to be updated are determined by the optimizer.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model to train.
        loader (DataLoader): DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): Instantiated optimizer to use.
        loss_function (torch.nn.Module): Loss function to use.

    Raises:
        ValueError: If the loss of a batch is not finite (``NaN`` or infinite). The error is raised before the parameters
            are updated with that batch.
    """
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        
        loss = _compute_loss(model, loss_function, pred, batch_y)
        _check_finite_loss(loss)
        loss.backward()
        
        optimizer.step()
