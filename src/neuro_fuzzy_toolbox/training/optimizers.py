import torch


class NormalizedGradientDescent(torch.optim.Optimizer):
    """
    Gradient descent with a step normalized by the norm of the whole gradient.

    At each step, every parameter is updated as

    .. math::

        \\eta \\leftarrow \\eta - \\alpha \\frac{\\partial E}{\\partial \\eta}, \\qquad
        \\alpha = \\frac{\\gamma}{\\sqrt{\\sum_{\\eta} \\left( \\frac{\\partial E}{\\partial \\eta} \\right)^2}}

    where the sum runs over **all** the parameters handled by the optimizer (across every parameter group), so that the
    Euclidean length of the whole update is exactly :math:`\\gamma`. This is the premise update rule of the original
    SONFIS formulation (Allende-Cid et al., 2016, eq. 8-10), whose step size satisfies :math:`0 < \\gamma < 1`.

    Note:
        Because the step is normalized, the scale of the loss does not affect the update: the mean squared error and
        the sum of squared errors produce the same step. To reproduce the original formulation, where the gradient is
        computed over the whole training set, the training DataLoader should contain a single batch.

    Args:
        params (iterable): Iterable of parameters (or dicts defining parameter groups) to optimize.
        gamma (float): Length of the update step, :math:`\\gamma`. Defaults to ``0.1``.
        eps (float): Gradient norm below which no update is performed. Defaults to ``1e-12``.
    """
    def __init__(self, params, gamma=0.1, eps=1e-12):
        if gamma <= 0.:
            raise ValueError(f'gamma must be positive, but received {gamma}.')
        if eps < 0.:
            raise ValueError(f'eps must be non-negative, but received {eps}.')
        super().__init__(params, dict(gamma=gamma, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss. Defaults to ``None``.

        Returns:
            torch.Tensor | None: The loss returned by ``closure``, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grads = [p.grad for group in self.param_groups for p in group['params'] if p.grad is not None]
        if len(grads) == 0:
            return loss

        # Norm of the whole gradient (all the parameters of all the groups)
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().to(torch.float64)) for g in grads]))

        for group in self.param_groups:
            if norm <= group['eps']:
                continue
            scale = group['gamma'] / norm.item()
            for p in group['params']:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-scale)
        return loss
