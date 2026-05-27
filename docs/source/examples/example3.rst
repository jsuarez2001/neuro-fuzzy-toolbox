.. _example3:

Example 3: Binary Classification on Heart Disease Dataset using SONFIS
========================================================================

This example demonstrates structural adaptation with
:ref:`SONFIS <sonfis-usage>` on the
`Heart Disease dataset <https://archive.ics.uci.edu/dataset/45/heart+disease>`_,
a 13-feature binary classification benchmark. The target variable is
binarized — distinguishing the presence from the absence of heart disease —
and a ``rule_reduced_ANFIS`` model is trained from a small initial rule base
that SONFIS then adapts through rule growing, splitting, and pruning.

This example also illustrates the use of ``lse_for_new_consequents=True``,
which initializes the consequent parameters of newly created rules using
least-squares estimation rather than random initialization, providing a
better starting point for subsequent gradient-based updates.

Imports and reproducibility
----------------------------
.. code-block:: python

    from ucimlrepo import fetch_ucirepo

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import (
        confusion_matrix, f1_score, precision_score,
        recall_score, accuracy_score, classification_report
    )

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

Data
----
Missing values are filled with zero. The original five-class target is
binarized: any value greater than 0 is mapped to 1, representing the
presence of heart disease. The dataset is split into training (70%),
validation (16%), and test (14%) sets using stratified sampling. Features
are scaled to [0, 1] and converted to ``torch.float64`` tensors to improve
numerical stability in the least-squares estimation steps.

.. code-block:: python

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Fill missing values
    X = X.fillna(value=0)

    # Convert to binary classification: 0 = no disease, 1 = disease
    y = y.copy()
    y.loc[y['num'] > 0, 'num'] = 1

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    scaler = MinMaxScaler(feature_range=(0, 1))

    x_train = torch.from_numpy(scaler.fit_transform(x_train)).to(torch.float64)
    x_val   = torch.from_numpy(scaler.transform(x_val)).to(torch.float64)
    x_test  = torch.from_numpy(scaler.transform(x_test)).to(torch.float64)

    y_train = torch.from_numpy(y_train.values).squeeze()
    y_val   = torch.from_numpy(y_val.values).squeeze()
    y_test  = torch.from_numpy(y_test.values).squeeze()

DataLoaders
-----------
.. code-block:: python

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = data.DataLoader(
        data.TensorDataset(x_train, y_train),
        batch_size=16, shuffle=True, generator=generator
    )
    val_loader = data.DataLoader(
        data.TensorDataset(x_val, y_val),
        batch_size=16, shuffle=False
    )

Model
-----
A ``rule_reduced_ANFIS`` model is instantiated with 3 initial rules,
``Gaussian_MF`` membership functions, and a softmax output layer for binary
classification. Premise parameters are initialized from the training data,
and consequent parameters are estimated by regularized least squares.

.. code-block:: python

    features = heart_disease.variables['name'][:13].tolist()

    model = nft.rule_reduced_ANFIS(
        input_size=x_train.shape[1],
        num_mfs=3,
        outputs=2,
        membership_function=nft.Gaussian_MF(),
        output_type='softmax',
        features=features,
        dtype=torch.float64
    )

    model.init_premises(x_train)
    model.init_consequents(x_train, y_train, driver='gelsd', ridge_lambda=1e-3)

Learning algorithm
------------------
The parameter update algorithm is defined here and passed to SONFIS as its
``ANFIStrainer``. It will be used internally by SONFIS to update the model
parameters at each structural adaptation iteration.

.. code-block:: python

    anfis_trainer = nft.Basic_optimizer_training_algorithm(
        epochs=1000,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW,
        optimizer_params={'lr': 1e-3, 'weight_decay': 1e-2},
        early_stopping=nft.EarlyStopping(patience=80)
    )

SONFIS
------
SONFIS is configured with rule growing, splitting, and pruning thresholds
appropriate for this dataset. Enabling ``lse_for_new_consequents`` ensures
that the consequent parameters of any rule added by GrowNet or SplitSubNet
are initialized via least-squares estimation rather than randomly, which
tends to reduce the number of gradient updates needed to integrate the new
rule into the model. A separate early stopping mechanism is provided at the
SONFIS iteration level, independent of the one used by the ``ANFIStrainer``.

.. code-block:: python

    sonfis = nft.SONFIS(
        Ngrow=20,
        dGrow=0.8,
        Nsplit=25,
        eSplit=0.35,
        Nvanish=5,
        lVanish=4,
        max_iterations=100,
        ANFIStrainer=anfis_trainer,
        early_stopping=nft.EarlyStopping(patience=25),
        lse_for_new_consequents=True,
        lse_for_new_consequents_lambda=1e-1,
        last_training_iteration=False
    )

    sonfis(model, train_loader, val_loader)

Evaluation
----------
.. code-block:: python

    pred = model.predict(x_test)

    acc        = accuracy_score(y_test, pred)
    prec       = precision_score(y_test, pred, zero_division=0)
    recall     = recall_score(y_test, pred)
    f1         = f1_score(y_test, pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, pred)
    class_rep  = classification_report(y_test, pred)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", recall)
    print("F1 score:", f1, "\n")

    print("Confusion Matrix:")
    print(conf_matrix, "\n")

    print("Classification Report:")
    print(class_rep)

.. code-block:: text

    Accuracy: 0.8131868131868132
    Precision: 0.8378378378378378
    Recall: 0.7380952380952381
    F1 score: 0.7848101265822784 

    Confusion Matrix:
    [[43  6]
     [11 31]] 

    Classification Report:
                  precision    recall  f1-score   support

               0       0.80      0.88      0.83        49
               1       0.84      0.74      0.78        42

        accuracy                           0.81        91
       macro avg       0.82      0.81      0.81        91
    weighted avg       0.82      0.81      0.81        91

.. code-block:: python

    print(model.rules)

.. code-block:: text

    6

.. note::
    The SONFIS parameters (``Ngrow``, ``dGrow``, ``Nsplit``, ``eSplit``,
    ``Nvanish``, ``lVanish``) control the structural adaptation operators
    and are dataset-dependent. For a detailed description of each parameter,
    see :ref:`SONFIS <sonfis-usage>`.