import torch
import torch.nn as nn

from ._loader_utils import get_loader_tensors

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
    
    if freezed_subnets is None:
        freezed_subnets = torch.zeros(ANFISmodel.rules, dtype=torch.bool)
    freezed_subnets = freezed_subnets.bool()
    active_subnets = ~freezed_subnets
    n_active = int(active_subnets.sum())
    
    # Least squares problem construction (only the columns of the active rules)
    w_norm = ANFISmodel.get_firing_levels(x, normalized=True)
    xe = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    fs = w_norm[:, active_subnets].unsqueeze(2).repeat(1, 1, xe.shape[1]).view(w_norm.shape[0], -1)
    X = xe.repeat(1, n_active)
        
    '''preliminary fix for the dtype issue'''
    if ANFISmodel._output_type == 'softmax':
        y = y.to(torch.int64)
        if ANFISmodel._custom_classes:
            y = torch.searchsorted(ANFISmodel.classes, y)
        y = torch.nn.functional.one_hot(y, ANFISmodel._outputs)
    if y.dtype != X.dtype:
        y = y.to(X.dtype)
    '''preliminary fix for the dtype issue'''
    
    # The frozen rules keep their consequents: their contribution is subtracted from the targets
    current_consequents = ANFISmodel.get_consequents()
    if freezed_subnets.any():
        frozen_contribution = torch.einsum('nk,nd,okd->no', w_norm[:, freezed_subnets], xe,
                                           current_consequents[:, freezed_subnets, :].to(xe.dtype)) # (n_samples, outputs)
        y = y - (frozen_contribution[:, 0] if y.dim() == 1 else frozen_contribution)
    
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
    new_active_consequents = C.t().reshape(ANFISmodel._outputs, n_active, xe.shape[1])
    
    if not freezed_subnets.any():
        return new_active_consequents
    
    new_consequents = current_consequents.clone()
    new_consequents[:, active_subnets, :] = new_active_consequents.to(new_consequents.dtype)
    return new_consequents


def _local_consequents_estimation(ANFISmodel, x, y, driver, ridge_lambda, freezed_subnets=None):
    """
    Estimates the consequent parameters of each rule separately, by weighted least squares, from input and target tensors.
    
    See :func:`local_consequents_estimation_with_WLS` for details. This function is shared by the training algorithms
    and by the ``init_consequents`` method of the models.
    
    Args:
        ANFISmodel (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model whose consequent parameters are to be estimated.
        x (torch.Tensor): Input data of shape ``(n_samples, input_size)``.
        y (torch.Tensor): Targets of shape ``(n_samples,)`` or ``(n_samples, outputs)``. For ``output_type='softmax'``,
            the targets are class labels.
        driver (str): Backend function to use for the least-squares estimation.
        ridge_lambda (float): Lambda value for Ridge regularization. If ``0.``, no regularization is applied.
        freezed_subnets (torch.Tensor, optional): Boolean tensor of length ``num_rules`` indicating which rules keep their
            current consequent parameters. Defaults to ``None``.
    
    Returns:
        torch.Tensor: Tensor containing the new consequent parameters, of shape ``(outputs, rules, input_size + 1)``.
    """
    n_rules = ANFISmodel.rules
    if freezed_subnets is None:
        freezed_subnets = torch.zeros(n_rules, dtype=torch.bool)
    active_idx = torch.where(~freezed_subnets.bool())[0]
    
    current_consequents = ANFISmodel.get_consequents()
    if active_idx.numel() == 0:
        return current_consequents
    
    xe = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1) # (n_samples, input_size + 1)
    
    '''preliminary fix for the dtype issue'''
    if ANFISmodel._output_type == 'softmax':
        y = y.to(torch.int64)
        if ANFISmodel._custom_classes:
            y = torch.searchsorted(ANFISmodel.classes, y)
        y = torch.nn.functional.one_hot(y, ANFISmodel._outputs)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    y = y.to(xe.dtype) # (n_samples, outputs)
    '''preliminary fix for the dtype issue'''
    
    # Unnormalized firing levels of the active rules (the default rule, if any, is excluded): (n_active, n_samples)
    with torch.no_grad():
        w = ANFISmodel.get_firing_levels(x)[:, :n_rules][:, active_idx].t().to(xe.dtype)
    
    # Each rule's weights are divided by their maximum. This does not change the weighted least-squares solution,
    # but keeps the Ridge penalty on a comparable scale regardless of the magnitude of the firing levels.
    max_w = w.max(dim=1, keepdim=True).values
    covered = max_w.squeeze(1) > 0 # rules that are activated by at least one sample
    w = torch.where(max_w > 0, w / torch.where(max_w > 0, max_w, torch.ones_like(max_w)), torch.zeros_like(w))
    
    sqrt_w = w.sqrt().unsqueeze(2)    # (n_active, n_samples, 1)
    A = sqrt_w * xe.unsqueeze(0)      # (n_active, n_samples, input_size + 1)
    B = sqrt_w * y.unsqueeze(0)       # (n_active, n_samples, outputs)
    
    if ridge_lambda > 0.:
        p = A.shape[2]
        I = torch.eye(p, dtype=A.dtype) * torch.sqrt(torch.tensor(ridge_lambda, dtype=A.dtype))
        A = torch.cat([A, I.unsqueeze(0).expand(A.shape[0], -1, -1)], dim=1)
        B = torch.cat([B, torch.zeros((B.shape[0], p, B.shape[2]), dtype=B.dtype)], dim=1)
    
    # One independent (weighted) least-squares problem per active rule, solved as a batch
    C = torch.linalg.lstsq(A, B, driver=driver).solution # (n_active, input_size + 1, outputs)
    
    new_consequents = current_consequents.clone()
    new_consequents[:, active_idx[covered], :] = C[covered].permute(2, 0, 1).to(new_consequents.dtype)
    return new_consequents


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
    """
    for batch_x, batch_y in loader:
        batch_y_copy = batch_y.clone().detach()
        
        '''preliminary fix for the dtype issue'''
        if not isinstance(loss_function, nn.CrossEntropyLoss): #cross_entropy function only accepts torch.long (torch.int64) dtype for target indices
            if batch_x.dtype != batch_y.dtype:
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