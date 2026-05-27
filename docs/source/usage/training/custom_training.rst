.. _custom-training:

Custom Training
===============

Neuro-Fuzzy Toolbox is built on top of PyTorch, so all model classes inherit
from ``nn.Module`` and are fully compatible with the standard PyTorch training
workflow. The built-in training algorithms described in the previous sections
cover the most common use cases, but the toolbox also exposes a lower-level
API for users who need finer control.

This API supports two main use cases:

- **Custom training procedures**: Instantiate PyTorch optimizers over
  arbitrary subsets of the model's parameters — all parameters jointly,
  premises only, consequents only, or individual rule parameters — and
  combine them with structural modification operations to implement
  problem-specific training strategies.
- **Deep neuro-fuzzy architectures**: Use the implemented models as
  differentiable components within larger PyTorch pipelines, connecting them
  to other ``nn.Module`` layers and training the resulting architecture
  end-to-end.

1. Accessing parameters for optimizers
---------------------------------------
PyTorch optimizers require an iterable of parameters to optimize. The toolbox
provides several methods to retrieve the model parameters at different levels
of granularity.

All parameters
^^^^^^^^^^^^^^
The standard ``model.parameters()`` method, inherited from ``nn.Module``,
returns all trainable parameters of the model. This is the typical entry point
for single-optimizer training:

.. code-block:: python

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

Premises and consequents separately
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The toolbox provides dedicated methods to retrieve the premise and consequent
parameters independently, enabling separate optimizers with different
hyperparameters for each parameter set:

.. code-block:: python

    prems_optimizer = torch.optim.AdamW(
        model.get_premises_as_parameters_list(),
        lr=1e-3, weight_decay=1e-2
    )

    cons_optimizer = torch.optim.AdamW(
        model.get_consequents_as_parameters_list(),
        lr=1e-3, weight_decay=1e-2
    )

This is exactly the approach used internally by the
:ref:`Double Optimizer Training Algorithm <double_optimizer>`.

.. note::
    The structure returned by ``get_premises_as_parameters_list()`` and
    ``get_consequents_as_parameters_list()`` differs between model classes,
    as described in :ref:`ANFIS Variants <anfis-variants-usage>`. However,
    both methods return an iterable compatible with PyTorch optimizers
    regardless of the underlying class.

Rule-specific access (``rule_reduced_ANFIS``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For ``rule_reduced_ANFIS`` models, the premise and consequent parameters are
stored per rule, so individual rule parameters can be accessed by indexing
into the lists returned by these methods. This makes it possible to
instantiate an optimizer over the parameters of a single rule in isolation:

.. code-block:: python

    # Optimizer over the last rule's parameters only
    rule_params = [
        model.get_premises_as_parameters_list()[-1],
        model.get_consequents_as_parameters_list()[-1]
    ]
    rule_optimizer = torch.optim.AdamW(rule_params, lr=5e-3, weight_decay=1e-2)

.. tip::
    This pattern is particularly useful when adding a new rule to an existing
    model: the new rule's parameters can be fine-tuned in isolation before
    committing to a global readaptation step, as shown in the
    :ref:`practical example <greedy-rule-growing-example>` below.

2. Structural modification
---------------------------
The ``rule_reduced_ANFIS`` class supports adding and removing rules at
runtime, enabling the model structure to be adapted dynamically during
training.

.. warning::
    The structural modification methods described in this section are
    **only available for** ``rule_reduced_ANFIS`` instances.

Adding rules
^^^^^^^^^^^^
The ``add_rules(means, stds)`` method adds one or more new rules to the
model. The premise parameters of the new rules are generated from the
provided means and standard deviations, following the initialization
convention of the chosen membership function. The consequent parameters
of the new rules are initialized randomly.

.. code-block:: python

    # Add a new rule centered on a specific training sample
    means = x_train[idx]                         # shape: (1, input_size)
    stds  = torch.full_like(means, 0.25)         # shape: (1, input_size)
    model.add_rules(means, stds)

    print(f"Number of rules after addition: {model.rules}")

The ``means`` and ``stds`` tensors must have shape
``(num_new_rules, input_size)``, so multiple rules can be added in a single
call by providing a batch of center and spread values.

Removing rules
^^^^^^^^^^^^^^
The ``remove_rules(rules_idxs)`` method removes the rules at the specified
indices:

.. code-block:: python

    # Remove the last rule
    model.remove_rules([model.rules - 1])

    print(f"Number of rules after removal: {model.rules}")

The argument is a list of integer indices, where each index must be in the
range ``[0, num_rules - 1]``.

3. Accessing intermediate layer outputs
-----------------------------------------
The toolbox provides methods to retrieve the outputs of intermediate layers,
which are useful both for implementing custom training logic and for
connecting a model to downstream components in a deep architecture.

Firing levels
^^^^^^^^^^^^^
The ``get_firing_levels(x, normalized=False)`` method returns the firing
levels produced by the T-norm layer for a given input batch. If
``normalized=True``, the normalized firing levels are returned instead:

.. code-block:: python

    # Unnormalized firing levels — shape: (batch_size, num_rules)
    firing_levels = model.get_firing_levels(x_train)

    # Normalized firing levels
    firing_levels_norm = model.get_firing_levels(x_train, normalized=True)

This is useful, for example, to identify which rules are most active for a
given set of samples — as done internally by the SONFIS algorithm to drive
its structural adaptation operators.

Individual rule outputs
^^^^^^^^^^^^^^^^^^^^^^^^
The ``get_all_consequents_outputs(x, weighted=True)`` method returns the
individual output of each rule for a given input batch. If ``weighted=True``
(the default), the outputs are weighted by the corresponding normalized firing
levels; otherwise, the raw unweighted rule outputs are returned:

.. code-block:: python

    # Weighted rule outputs — shape: (outputs, batch_size, num_rules)
    rule_outputs = model.get_all_consequents_outputs(x_train)

    # Unweighted rule outputs
    rule_outputs_raw = model.get_all_consequents_outputs(x_train, weighted=False)

In the context of deep neuro-fuzzy architectures, this method allows the
per-rule outputs to be used as intermediate features that can be passed to
subsequent layers of a larger network.

.. _greedy-rule-growing-example:

4. Practical example: greedy rule-growing
------------------------------------------
The following example combines the tools described above to implement a
simple greedy rule-growing procedure on the
`Glass Identification dataset <https://doi.org/10.24432/C5WW2P>`_,
a 9-feature, 6-class benchmark. Its dimensionality makes
``rule_reduced_ANFIS`` a more suitable choice than classical ANFIS.

The procedure starts from an already-trained model with an initial rule
base and iteratively attempts to grow it. At each step, a new rule is added
centered on a sample from the worst-performing class. The new rule's
parameters are then fine-tuned in isolation. If validation loss improves, the
rule is retained and a global readaptation step is performed; otherwise, the
rule is discarded. The procedure terminates when a maximum number of
consecutive failed attempts is reached.

Model and initial training
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    import neuro_fuzzy_toolbox as nft
    import torch
    import torch.nn as nn
    import torch.utils.data as data
    from sklearn.metrics import recall_score

    # Model definition
    model = nft.rule_reduced_ANFIS(
        input_size=x_train.shape[1],
        num_mfs=5,          # initial number of rules
        outputs=6,
        membership_function=nft.GeneralizedBell_MF(),
        output_type='softmax',
        features=features
    )

    # Initial training with a built-in algorithm
    trainer = nft.Basic_optimizer_training_algorithm(
        epochs=5000,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW,
        optimizer_params={'lr': 1e-3, 'weight_decay': 1e-2},
        early_stopping=nft.EarlyStopping(patience=60)
    )
    trainer(model, train_loader, val_loader)

Helper functions
^^^^^^^^^^^^^^^^^
.. code-block:: python

    loss_function = nn.CrossEntropyLoss()

    def val_loss(model):
        with torch.no_grad():
            return sum(
                loss_function(model(xb), yb)
                for xb, yb in val_loader
            ) / len(val_loader)

Greedy rule-growing loop
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # Hyperparameters
    max_failed_attempts     = 5
    single_adaptation_lr    = 0.005
    single_adaptation_epochs = 500
    single_patience         = 30
    global_adaptation_lr    = 0.001
    global_adaptation_epochs = 1000
    global_patience         = 60

    failed_attempts = 0
    best_loss = val_loss(model)

    while failed_attempts < max_failed_attempts:

        # Identify the worst-recall class
        with torch.no_grad():
            pred_train = model.predict(x_train)
        recalls = recall_score(
            y_train.numpy(), pred_train.numpy(),
            average=None, zero_division=0
        )
        worst_class = int(recalls.argmin())

        # Add a rule centered on a sample from the worst class
        class_indices = (y_train == worst_class).nonzero(as_tuple=True)[0]
        idx = class_indices[torch.randint(0, len(class_indices), (1,))]
        model.add_rules(
            x_train[idx].to(torch.float32),
            torch.full_like(x_train[idx], 0.25)
        )

        # Fine-tune only the new rule's parameters
        new_params = [
            model.get_premises_as_parameters_list()[-1],
            model.get_consequents_as_parameters_list()[-1]
        ]
        opt_new = torch.optim.AdamW(
            new_params, lr=single_adaptation_lr, weight_decay=0.01
        )
        best_single, patience_counter = val_loss(model), 0
        for epoch in range(single_adaptation_epochs):
            for xb, yb in train_loader:
                opt_new.zero_grad()
                loss_function(model(xb), yb).backward()
                opt_new.step()
            current = val_loss(model)
            patience_counter = 0 if current < best_single else patience_counter + 1
            best_single = min(best_single, current)
            if patience_counter >= single_patience:
                break

        # Retain or discard the new rule
        if val_loss(model) < best_loss:
            # Global readaptation of all parameters
            opt_all = torch.optim.AdamW(
                model.parameters(), lr=global_adaptation_lr, weight_decay=0.01
            )
            best_global, patience_counter = val_loss(model), 0
            for epoch in range(global_adaptation_epochs):
                for xb, yb in train_loader:
                    opt_all.zero_grad()
                    loss_function(model(xb), yb).backward()
                    opt_all.step()
                current = val_loss(model)
                patience_counter = 0 if current < best_global else patience_counter + 1
                best_global = min(best_global, current)
                if patience_counter >= global_patience:
                    break
            best_loss = val_loss(model)
            failed_attempts = 0
        else:
            model.remove_rules([model.rules - 1])
            failed_attempts += 1

    print(f"Final number of rules: {model.rules}")

.. note::
    Users who do not require a fully custom update scheme may rely on the
    built-in :ref:`SONFIS <sonfis-usage>` algorithm instead, which
    encapsulates rule growing, splitting, and pruning within a
    self-organizing training loop operating directly on
    ``rule_reduced_ANFIS`` models.