import torch
import torch.nn as nn



def consequentsUpdate(ANFISmodel, x_train, y_train, freezed=None):
    """
    Update the consequents of an ANFIS model based on input-output pairs.

    Parameters:
    - ANFISmodel (Type3ANFIS): An instance of the Type3ANFIS model.
    - x_train (torch.Tensor): Input training data.
    - y_train (torch.Tensor): Output training data.
    - freezed (torch.Tensor or None): Binary tensor indicating which rules of the model are freezed (default: None).

    Returns:
    - new_consequents (torch.Tensor): Updated consequent parameters.

    """
    new_consequents = torch.zeros_like(ANFISmodel.consequents)

    # Calculate normalized firing levels using the ANFIS model
    w_norm = ANFISmodel.norm_firing_levels(x_train)

    # Extend input data with a column of ones
    xe = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)

    # Check if any consequents are frozen
    if freezed is None:
        freezed = torch.zeros(new_consequents.shape[0])

    # Update each consequent individually
    for i in range(new_consequents.shape[0]):
        if freezed[i] == 0:
            # Repeat normalized firing levels to create a weight matrix
            w = w_norm[:, i].repeat(x_train.shape[1] + 1, 1)

            # Perform least squares solution to update the consequents
            new_consequents[i], _, _, _ = torch.linalg.lstsq(w @ xe, y_train)

    return new_consequents



def premisesUpdate(ANFISmodel, x_train, y_train, y=0.01, loss_function=nn.functional.mse_loss, freezed=None):
    """
    Update the premises (fuzzy sets) of an ANFIS model based on input-output pairs.

    Parameters:
    - ANFISmodel (Type3ANFIS): An instance of the Type3ANFIS model.
    - x_train (torch.Tensor): Input training data.
    - y_train (torch.Tensor): Output training data.
    - y (float): Learning rate for premises update (default: 0.01).
    - loss_function (torch.nn.Module): Loss function for training (default: nn.functional.mse_loss).
    - freezed (torch.Tensor or None): Binary tensor indicating which rules are freezed (default: None).

    Returns:
    - new_premises (torch.Tensor): Updated fuzzy premises.

    """
    # Forward pass to obtain model predictions
    pred = ANFISmodel(x_train)

    # Calculate loss and perform backward pass
    loss = loss_function(pred, y_train)
    loss.backward()

    # Calculate the step size alpha for the update
    alpha = y / torch.sqrt(torch.sum(torch.pow(ANFISmodel.fuzzify_layer.premises.grad, 2)))

    # Check if any rules are frozen
    if freezed is None:
        freezed = torch.zeros(ANFISmodel.rules)

    # Initialize new premises with the current premises
    new_premises = ANFISmodel.fuzzify_layer.premises

    # Extract parameters from the premises (mu and sigma)
    vs = new_premises[:, :, 0].t()
    sigmas = new_premises[:, :, 1].t()

    # Initialize tensors for the updated premises
    new_vs = torch.zeros_like(vs)
    new_sigmas = torch.zeros_like(sigmas)

    # Get normalized firing levels and outputs by rule
    w_norm, outputs_by_rule = ANFISmodel.intermediate_values(x_train)

    # Update premises by rule
    for k in range(ANFISmodel.rules):
        if freezed[k] == 0:
            A = 4 * alpha * (1 / torch.pow(sigmas[k], 2)) * (x_train - vs[k])
            B = 4 * alpha * (1 / torch.pow(sigmas[k], 3)) * torch.pow((x_train - vs[k]), 2)
            wk = w_norm[:, k].unsqueeze(0).t()
            fk = outputs_by_rule[:, k]
            zk = ((fk - pred) * (y_train - pred)).unsqueeze(0).t()

            # Update mu (vs) and sigma (sigmas) for each rule
            new_vs[k] = torch.sum(A * wk * zk, dim=0)
            new_sigmas[k] = torch.sum(B * wk * zk, dim=0)

    # Update the premises with the new values
    new_premises[:, :, 0] += new_vs.t()
    new_premises[:, :, 1] += new_sigmas.t()

    return new_premises
