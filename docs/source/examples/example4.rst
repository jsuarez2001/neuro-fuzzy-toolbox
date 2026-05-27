.. _example4:

Example 4: Regression on a 3D Surface using SONFIS
====================================================

This example demonstrates structural adaptation with
:ref:`SONFIS <sonfis-usage>` on a regression problem. The target is a
noisy two-input, one-output function with multiple peaks and valleys — a
common benchmark for nonlinear function approximation. A
``rule_reduced_ANFIS`` model is trained from a small initial rule base that
SONFIS adapts through rule growing, splitting, and pruning.

The example also shows how to visualize the target surface and the model
predictions side by side after training, which provides an intuitive picture
of how well the model has captured the underlying function.

Imports and reproducibility
----------------------------
.. code-block:: python

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import (
        mean_squared_error, root_mean_squared_error,
        mean_absolute_error, r2_score,
        mean_absolute_percentage_error
    )

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    import torch
    import torch.nn as nn
    import torch.utils.data as data
    import numpy as np
    import random

    import neuro_fuzzy_toolbox as nft

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

Visualization helper
---------------------
.. code-block:: python

    def plot_surface(X: torch.Tensor, Y: torch.Tensor, fig_size=(8, 6)):
        """
        Plots a 3D scatter surface from two input features and a target vector.

        Args:
            X (torch.Tensor): Input tensor of shape (n, 2).
            Y (torch.Tensor): Target tensor of shape (n,).
            fig_size (tuple): Figure size as (width, height) in inches.
        """
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()

        fig = plt.figure(figsize=fig_size)
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(X_np[:, 0], X_np[:, 1], Y_np, c=Y_np, cmap='viridis', s=1)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
        plt.tight_layout()
        plt.show()

Data
----
The target function is a two-variable peaks-like surface. Training samples
are generated with additive Gaussian noise, while the test set is noise-free
to evaluate the model's ability to recover the underlying function. Features
are scaled to [0, 1] and converted to ``torch.float64`` tensors.

.. code-block:: python

    # Target function (peaks-like 3D surface)
    def z(x, y):
        return (
            3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2)
            - 10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2) - y**2)
            - (1 / 3) * np.exp(-(x + 1)**2 - y**2)
        )

    # Training data with additive Gaussian noise
    x0    = np.random.uniform(-3, 3, 1000)
    x1    = np.random.uniform(-3, 3, 1000)
    noise = np.random.normal(0, 0.5, 1000)
    y_noisy = z(x0, x1) + noise

    # Test data (noise-free)
    x0_test = np.random.uniform(-3, 3, 1000)
    x1_test = np.random.uniform(-3, 3, 1000)
    y_clean = z(x0_test, x1_test)

    X_train_np = np.vstack((x0, x1)).T
    X_test_np  = np.vstack((x0_test, x1_test)).T

    x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(
        X_train_np, y_noisy, test_size=0.2, random_state=SEED
    )

    dtype = torch.float64

    scaler  = MinMaxScaler(feature_range=(0, 1))

    x_train = torch.tensor(scaler.fit_transform(x_train_np), dtype=dtype)
    x_val   = torch.tensor(scaler.transform(x_val_np),       dtype=dtype)
    x_test  = torch.tensor(scaler.transform(X_test_np),      dtype=dtype)

    y_train = torch.tensor(y_train_np, dtype=dtype)
    y_val   = torch.tensor(y_val_np,   dtype=dtype)
    y_test  = torch.tensor(y_clean,    dtype=dtype)

    # Visualize the test surface
    plot_surface(x_test, y_test)

DataLoaders
-----------
.. code-block:: python

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = data.DataLoader(
        data.TensorDataset(x_train, y_train),
        batch_size=64, shuffle=True, generator=generator
    )
    val_loader = data.DataLoader(
        data.TensorDataset(x_val, y_val),
        batch_size=64, shuffle=False
    )

Model
-----
A ``rule_reduced_ANFIS`` model is instantiated with 3 initial rules,
``GeneralizedBell_MF`` membership functions, and a default (regression)
output layer. Premise parameters are initialized from the training data,
and consequent parameters are estimated by regularized least squares.

.. code-block:: python

    model = nft.rule_reduced_ANFIS(
        input_size=2,
        num_mfs=3,
        outputs=1,
        membership_function=nft.GeneralizedBell_MF(),
        output_type='default',
        dtype=dtype
    )

    model.init_premises(x_train)
    model.init_consequents(x_train, y_train, driver='gelsd', ridge_lambda=1.0)

Learning algorithm
------------------
The parameter update algorithm is defined here and passed to SONFIS as
its ``ANFIStrainer``. Ridge regularization in the least-squares
initialization is set to a larger value (``1.0``) than in the classification
examples, reflecting the wider output range of the regression target.

.. code-block:: python

    anfis_trainer = nft.Basic_optimizer_training_algorithm(
        epochs=500,
        loss_function=nn.MSELoss(),
        optimizer=torch.optim.AdamW,
        optimizer_params={'lr': 5e-3, 'weight_decay': 1e-2},
        early_stopping=nft.EarlyStopping(patience=30, delta=1e-3)
    )

SONFIS
------
The structural adaptation thresholds are tuned for the regression setting.
``Ngrow`` and ``Nsplit`` are set higher than in the classification examples
to account for the larger training set and the more complex target function.
Setting ``last_training_iteration=True`` performs a final global parameter
update over all subnets after the structural adaptation has converged,
allowing the model to readapt its parameters to the final rule structure.

.. code-block:: python

    sonfis = nft.SONFIS(
        Ngrow=100,
        dGrow=0.8,
        Nsplit=140,
        eSplit=0.15,
        Nvanish=10,
        lVanish=3,
        max_iterations=100,
        ANFIStrainer=anfis_trainer,
        early_stopping=nft.EarlyStopping(patience=15),
        lse_for_new_consequents=True,
        lse_for_new_consequents_lambda=1e-1,
        last_training_iteration=True
    )

    sonfis(model, train_loader, val_loader)

Evaluation
----------
.. code-block:: python

    pred = model.predict(x_test)

    mse  = mean_squared_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)

    print("MSE: ", mse)
    print("RMSE:", rmse)
    print("MAE: ", mae)
    print("R²:  ", r2)
    print("MAPE:", mape)

.. code-block:: text

    MSE:  0.18704400823769493
    RMSE: 0.43248584744208096
    MAE:  0.2957790657497346
    R2:   0.9462171057884828
    MAPE: 40.81359011343615

.. code-block:: python

    print(model.rules)

.. code-block:: text

    12

The trained model's predictions can be compared visually against the
noise-free test surface:

.. code-block:: python

    print("Test surface:")
    plot_surface(x_test, y_test)

    print("Model predictions:")
    plot_surface(x_test, pred)

.. note::
    The SONFIS parameters (``Ngrow``, ``dGrow``, ``Nsplit``, ``eSplit``,
    ``Nvanish``, ``lVanish``) control the structural adaptation operators
    and are dataset-dependent. For a detailed description of each parameter,
    see :ref:`SONFIS <sonfis-usage>`.