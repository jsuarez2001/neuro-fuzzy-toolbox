"""
Least-squares estimation of the consequent parameters of ANFIS models.

Every least-squares estimation of the toolbox is built from the functions of this module, which are organized in three
levels:

- **Primitives**, which know nothing about the models: the extended regressor matrix, the Ridge augmentation and the
  solver.
- **Problems**, which also know nothing about the models: the joint problem, in which the consequents of a group of
  rules are estimated together from their normalized firing levels, and the weighted problem, in which the consequents
  of each rule are estimated separately, weighted by its firing level.
- **Model-level estimations**, which read the firing levels and the targets of a model and solve one of the two
  problems. They only use the public methods of the model (and a few attributes describing its outputs), so this module
  does not import any other package of the toolbox.

Every tensor created here is placed on the device of the input data.
"""
import torch


# ----------------------------------------------------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------------------------------------------------

def _extended_inputs(x):
    """
    Returns the regressor matrix extended by a column of ones, :math:`[\\mathbf{x}, 1]`.

    Args:
        x (torch.Tensor): Input data of shape ``(n_samples, input_size)``.

    Returns:
        torch.Tensor: Tensor of shape ``(n_samples, input_size + 1)``, with the dtype and device of ``x``.
    """
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=1)


def _ridge_augment(A, B, ridge_lambda):
    """
    Augments a least-squares system so that its solution is the Ridge solution.

    Appends the rows :math:`\\sqrt{\\lambda} I` to ``A`` and as many rows of zeros to ``B``, which turns the problem
    :math:`\\min \\lVert A C - B \\rVert^2 + \\lambda \\lVert C \\rVert^2` into an ordinary least-squares problem.
    Leading batch dimensions are supported.

    Args:
        A (torch.Tensor): Design matrix of shape ``(..., n_samples, n_params)``.
        B (torch.Tensor): Targets of shape ``(..., n_samples, n_targets)``.
        ridge_lambda (float): Ridge coefficient. If ``0.``, the system is returned unchanged.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The augmented ``A`` and ``B``.
    """
    if ridge_lambda <= 0.:
        return A, B
    p = A.shape[-1]
    I = torch.eye(p, dtype=A.dtype, device=A.device) * torch.sqrt(torch.tensor(ridge_lambda, dtype=A.dtype, device=A.device))
    I = I.expand(*A.shape[:-2], p, p)
    zeros = torch.zeros((*B.shape[:-2], p, B.shape[-1]), dtype=B.dtype, device=B.device)
    return torch.cat([A, I], dim=-2), torch.cat([B, zeros], dim=-2)


def _solve(A, B, driver):
    """
    Solves the (possibly batched) least-squares problem :math:`\\min_C \\lVert A C - B \\rVert^2`.

    Args:
        A (torch.Tensor): Design matrix of shape ``(..., n_samples, n_params)``.
        B (torch.Tensor): Targets of shape ``(..., n_samples, n_targets)``.
        driver (str): Backend of :func:`torch.linalg.lstsq` (``'gels'``, ``'gelsy'``, ``'gelsd'`` or ``'gelss'``).
            If ``None``, the default of :func:`torch.linalg.lstsq` is used.

    Returns:
        torch.Tensor: Solution of shape ``(..., n_params, n_targets)``.
    """
    return torch.linalg.lstsq(A, B.to(A.dtype), driver=driver).solution


# ----------------------------------------------------------------------------------------------------------------------
# Problems
# ----------------------------------------------------------------------------------------------------------------------

def _joint_least_squares(w_norm, xe, Y, ridge_lambda, driver):
    """
    Estimates jointly the consequents of a group of rules.

    Solves

    .. math::

        \\min_{\\{\\Theta_k\\}} \\sum_{t} \\left\\lVert \\mathbf{y}_t - \\sum_{k} \\bar{w}_k(\\mathbf{x}_t)\\,
        \\Theta_k \\mathbf{x}_{e,t} \\right\\rVert^2 + \\lambda \\sum_k \\lVert \\Theta_k \\rVert^2,

    where the sum over :math:`k` runs over the rules of the group.

    Args:
        w_norm (torch.Tensor): Normalized firing levels of the rules of the group, of shape ``(n_samples, n_rules)``.
        xe (torch.Tensor): Extended regressor matrix, of shape ``(n_samples, input_size + 1)``.
        Y (torch.Tensor): Targets, of shape ``(n_samples, outputs)``.
        ridge_lambda (float): Ridge coefficient. If ``0.``, no regularization is applied.
        driver (str): Backend of :func:`torch.linalg.lstsq`.

    Returns:
        torch.Tensor: Consequent parameters of the group, of shape ``(outputs, n_rules, input_size + 1)``.
    """
    n_samples, n_rules = w_norm.shape
    p = xe.shape[1]
    # Column k * p + j of the design matrix is w_k * xe_j
    A = (w_norm.unsqueeze(2) * xe.unsqueeze(1)).reshape(n_samples, n_rules * p)
    A, B = _ridge_augment(A, Y.to(A.dtype), ridge_lambda)
    C = _solve(A, B, driver)                    # (n_rules * p, outputs)
    return C.t().reshape(Y.shape[1], n_rules, p)


def _weighted_least_squares(w, xe, Y, ridge_lambda, driver):
    """
    Estimates separately the consequents of each rule, by weighted least squares.

    For each rule :math:`k`, solves

    .. math::

        \\min_{\\Theta_k} \\sum_{t} w_k(\\mathbf{x}_t) \\left\\lVert \\mathbf{y}_t - \\Theta_k \\mathbf{x}_{e,t}
        \\right\\rVert^2 + \\lambda \\lVert \\Theta_k \\rVert^2.

    All the problems are solved as a single batch.

    Args:
        w (torch.Tensor): Non-negative weights of each rule, of shape ``(n_rules, n_samples)``.
        xe (torch.Tensor): Extended regressor matrix, of shape ``(n_samples, input_size + 1)``.
        Y (torch.Tensor): Targets, of shape ``(n_samples, outputs)``.
        ridge_lambda (float): Ridge coefficient. If ``0.``, no regularization is applied.
        driver (str): Backend of :func:`torch.linalg.lstsq`.

    Returns:
        torch.Tensor: Consequent parameters, of shape ``(n_rules, input_size + 1, outputs)``.
    """
    sqrt_w = w.sqrt().unsqueeze(2)    # (n_rules, n_samples, 1)
    A = sqrt_w * xe.unsqueeze(0)      # (n_rules, n_samples, input_size + 1)
    B = sqrt_w * Y.unsqueeze(0)       # (n_rules, n_samples, outputs)
    A, B = _ridge_augment(A, B, ridge_lambda)
    return _solve(A, B, driver)


# ----------------------------------------------------------------------------------------------------------------------
# Model-level estimations
# ----------------------------------------------------------------------------------------------------------------------

def _encode_targets(model, y, dtype):
    """
    Converts the targets of a model into the matrix used by the least-squares problems.

    For ``output_type='softmax'``, the class labels are mapped to their indices (if the model uses custom class labels)
    and one-hot encoded. Otherwise the targets are used as they are.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model.
        y (torch.Tensor): Targets of shape ``(n_samples,)`` or ``(n_samples, outputs)``, or class labels.
        dtype (torch.dtype): Dtype of the result.

    Returns:
        torch.Tensor: Targets of shape ``(n_samples, outputs)``.
    """
    if model._output_type == 'softmax':
        y = y.to(torch.int64)
        if model._custom_classes:
            y = torch.searchsorted(model.classes, y)
        y = torch.nn.functional.one_hot(y, model._outputs)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    return y.to(dtype)


def _global_consequents_estimation(model, x, y, driver, ridge_lambda, freezed_subnets=None):
    """
    Estimates jointly the consequent parameters of all the rules of a model (global estimation).

    If ``freezed_subnets`` is provided, only the rules that are not frozen are estimated, on the residual left by the
    frozen ones, which keep their current consequents. See
    :func:`~neuro_fuzzy_toolbox.training.update_strategies.classical_consequents_estimation_with_OLS`.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model.
        x (torch.Tensor): Input data of shape ``(n_samples, input_size)``.
        y (torch.Tensor): Targets of shape ``(n_samples,)`` or ``(n_samples, outputs)``, or class labels.
        driver (str): Backend of :func:`torch.linalg.lstsq`.
        ridge_lambda (float): Ridge coefficient. If ``0.``, no regularization is applied.
        freezed_subnets (torch.Tensor, optional): Boolean tensor of length ``num_rules`` indicating which rules keep
            their current consequent parameters. Defaults to ``None``.

    Returns:
        torch.Tensor: Consequent parameters of shape ``(outputs, rules, input_size + 1)``.
    """
    if freezed_subnets is None:
        freezed_subnets = torch.zeros(model.rules, dtype=torch.bool)
    freezed_subnets = freezed_subnets.bool()
    active_subnets = ~freezed_subnets

    w_norm = model.get_firing_levels(x, normalized=True)
    xe = _extended_inputs(x)
    Y = _encode_targets(model, y, xe.dtype)

    # The frozen rules keep their consequents: their contribution is subtracted from the targets
    current_consequents = model.get_consequents()
    if freezed_subnets.any():
        Y = Y - torch.einsum('nk,nd,okd->no', w_norm[:, freezed_subnets], xe,
                             current_consequents[:, freezed_subnets, :].to(xe.dtype))

    new_active_consequents = _joint_least_squares(w_norm[:, active_subnets], xe, Y, ridge_lambda, driver)
    if not freezed_subnets.any():
        return new_active_consequents

    new_consequents = current_consequents.clone()
    new_consequents[:, active_subnets, :] = new_active_consequents.to(new_consequents.dtype)
    return new_consequents


def _local_consequents_estimation(model, x, y, driver, ridge_lambda, freezed_subnets=None):
    """
    Estimates separately the consequent parameters of each rule of a model, by weighted least squares (local estimation).

    See :func:`~neuro_fuzzy_toolbox.training.update_strategies.local_consequents_estimation_with_WLS`. The unnormalized
    firing levels of each rule are divided by their maximum before solving, which does not change the solution but keeps
    the Ridge penalty on a comparable scale across rules. The default rule, if any, is not estimated, and the rules that
    are not activated by any sample keep their current consequents.

    Args:
        model (ANFIS | h_ANFIS | rule_reduced_ANFIS): ANFIS model.
        x (torch.Tensor): Input data of shape ``(n_samples, input_size)``.
        y (torch.Tensor): Targets of shape ``(n_samples,)`` or ``(n_samples, outputs)``, or class labels.
        driver (str): Backend of :func:`torch.linalg.lstsq`.
        ridge_lambda (float): Ridge coefficient. If ``0.``, no regularization is applied.
        freezed_subnets (torch.Tensor, optional): Boolean tensor of length ``num_rules`` indicating which rules keep
            their current consequent parameters. Defaults to ``None``.

    Returns:
        torch.Tensor: Consequent parameters of shape ``(outputs, rules, input_size + 1)``.
    """
    n_rules = model.rules
    if freezed_subnets is None:
        freezed_subnets = torch.zeros(n_rules, dtype=torch.bool)
    active_idx = torch.where(~freezed_subnets.bool())[0]

    current_consequents = model.get_consequents()
    if active_idx.numel() == 0:
        return current_consequents

    xe = _extended_inputs(x)
    Y = _encode_targets(model, y, xe.dtype)

    # Unnormalized firing levels of the active rules (the default rule, if any, is excluded): (n_active, n_samples)
    w = model.get_firing_levels(x)[:, :n_rules][:, active_idx].t().to(xe.dtype)

    max_w = w.max(dim=1, keepdim=True).values
    covered = max_w.squeeze(1) > 0 # rules that are activated by at least one sample
    w = torch.where(max_w > 0, w / torch.where(max_w > 0, max_w, torch.ones_like(max_w)), torch.zeros_like(w))

    C = _weighted_least_squares(w, xe, Y, ridge_lambda, driver) # (n_active, input_size + 1, outputs)

    new_consequents = current_consequents.clone()
    new_consequents[:, active_idx[covered], :] = C[covered].permute(2, 0, 1).to(new_consequents.dtype)
    return new_consequents


def _rules_consequents_estimation(model, x, y, rules, driver, ridge_lambda):
    """
    Estimates jointly the consequent parameters of some rules of a model, ignoring the contribution of the others.

    Used by SONFIS to initialize the consequents of the rules created by GrowNet and SplitNet from the samples that
    originated them (see ``lse_for_new_consequents``).

    Args:
        model (rule_reduced_ANFIS): ANFIS model.
        x (torch.Tensor): Input data of shape ``(n_samples, input_size)``.
        y (torch.Tensor): Targets of shape ``(n_samples,)`` or ``(n_samples, outputs)``, or class labels.
        rules (list[int]): Indices of the rules to estimate.
        driver (str): Backend of :func:`torch.linalg.lstsq`.
        ridge_lambda (float): Ridge coefficient. If ``0.``, no regularization is applied.

    Returns:
        torch.Tensor: Consequent parameters of the given rules, of shape ``(outputs, len(rules), input_size + 1)``.
    """
    w_norm = model.get_firing_levels(x, normalized=True)[:, rules]
    xe = _extended_inputs(x)
    Y = _encode_targets(model, y, xe.dtype)
    return _joint_least_squares(w_norm, xe, Y, ridge_lambda, driver)
