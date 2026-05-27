.. _double_optimizer:

Double Optimizer Training Algorithm
=====================================
A dual-optimizer gradient-based training algorithm for ANFIS models. Unlike
the :ref:`Basic Optimizer Training Algorithm <basic_optimizer>`, this
algorithm uses two separate optimizers to update the premise and consequent
parameters independently, allowing different optimizer classes and
hyperparameters to be specified for each parameter set.

The update strategy is controlled by the ``mode`` parameter:

- **Mode 0**: In each epoch, a single forward pass is performed and both
  the premise and consequent parameters are updated simultaneously using
  their respective optimizers.
- **Mode 1**: Each epoch consists of two sequential passes through the
  data. In the first pass, only the consequent parameters are updated;
  in the second pass, only the premise parameters are updated.

.. note::
    - For more details on its implementation in the toolbox, see :ref:`Double Optimizer Training Algorithm`.

Instantiation
-------------
The following parameters are available when instantiating this training
algorithm:

- **epochs** (``int``): Number of training epochs.
- **loss_function** (``torch.nn.Module``): Instantiated loss function to use
  during training (e.g., ``torch.nn.CrossEntropyLoss()``).
- **early_stopping** (``nft.EarlyStopping``): Early stopping mechanism to use
  during training (Default: ``None``).
- **mode** (``int``): Update strategy to use. ``0`` updates premises and
  consequents simultaneously in a single pass per epoch; ``1`` updates them
  sequentially in two separate passes per epoch (Default: ``0``).
- **prems_optim** (``torch.optim.Optimizer``): Optimizer class to use for
  the premise parameters (Default: ``torch.optim.Adam``).
- **prems_optim_params** (``dict``): Parameters to pass to the premise
  optimizer (Default: ``{}``).
- **cons_optim** (``torch.optim.Optimizer``): Optimizer class to use for
  the consequent parameters (Default: ``torch.optim.Adam``).
- **cons_optim_params** (``dict``): Parameters to pass to the consequent
  optimizer (Default: ``{}``).

Example
^^^^^^^
Instantiating the algorithm with different optimizers and hyperparameters
for premises and consequents:

.. code-block:: python

    import neuro_fuzzy_toolbox as nft
    import torch
    import torch.nn as nn

    trainer = nft.Double_optimizer_training_algorithm(
        epochs=500,
        loss_function=nn.CrossEntropyLoss(),
        early_stopping=nft.EarlyStopping(patience=30, delta=1e-4),
        mode=0,
        prems_optim=torch.optim.AdamW,
        prems_optim_params={'lr': 1e-3, 'weight_decay': 1e-2},
        cons_optim=torch.optim.AdamW,
        cons_optim_params={'lr': 1e-3, 'weight_decay': 1e-2}
    )

The following arguments are available when calling the training algorithm
via ``__call__``:

- **model**: ANFIS model to train.
- **train_loader**: DataLoader with the training data.
- **val_loader**: DataLoader with the validation data (Default: ``None``).
- **verbose**: Whether to print progress messages (Default: ``True``).

Assuming ``model`` is an instantiated ANFIS model and ``train_loader``
and ``val_loader`` are PyTorch DataLoaders, training is invoked as follows:

.. code-block:: python

    trainer(model, train_loader, val_loader)

.. important::
    The training batch size is determined by the DataLoader, so this should
    be taken into account when defining it (see
    :ref:`PyTorch DataLoaders <DataLoaders_usage>`).