.. _example2:

Example 2: Multiclass Classification on Glass Identification Dataset
=====================================================================

This example demonstrates the low-level API of Neuro-Fuzzy Toolbox on the
`Glass Identification dataset <https://doi.org/10.24432/C5WW2P>`_, a
nine-feature, six-class classification benchmark. Its dimensionality makes
``rule_reduced_ANFIS`` a more suitable choice than classical ANFIS, since
it avoids the combinatorial growth of rules with the number of input features.

The example combines an initial gradient-based training phase with a custom
greedy rule-growing procedure that iteratively expands the rule base by
targeting the worst-performing class at each step. This is the same workflow
described in the :ref:`Custom Training <custom-training>` section.

Imports and reproducibility
----------------------------
.. code-block:: python

    from ucimlrepo import fetch_ucirepo

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
The class labels are re-encoded with ``LabelEncoder`` to produce contiguous
integer indices starting from 0, as required by ``CrossEntropyLoss``. The
dataset is split into training (70%), validation (16%), and test (14%) sets
using stratified sampling.

.. code-block:: python

    glass_identification = fetch_ucirepo(id=42)

    X = glass_identification.data.features
    y = glass_identification.data.targets

    le = LabelEncoder()
    y.loc[:, 'Type_of_glass'] = le.fit_transform(y['Type_of_glass'])
    y = y.astype('int64')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    scaler = MinMaxScaler(feature_range=(0, 1))

    x_train = torch.from_numpy(scaler.fit_transform(x_train)).to(torch.float32)
    x_val   = torch.from_numpy(scaler.transform(x_val)).to(torch.float32)
    x_test  = torch.from_numpy(scaler.transform(x_test)).to(torch.float32)

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
        batch_size=8, shuffle=True, generator=generator
    )
    val_loader = data.DataLoader(
        data.TensorDataset(x_val, y_val),
        batch_size=8, shuffle=False
    )

Model
-----
A ``rule_reduced_ANFIS`` model is instantiated with 5 initial rules,
``GeneralizedBell_MF`` membership functions, and a softmax output layer
for six-class classification. The custom rule-growing procedure will
expand the rule base dynamically during training.

.. code-block:: python

    features = X.columns.tolist()

    model = nft.rule_reduced_ANFIS(
        input_size=x_train.shape[1],
        num_mfs=5, # 5 rules initially (rule-reduced model)
        outputs=6,
        membership_function=nft.GeneralizedBell_MF(),
        output_type='softmax',
        features=features
    )

Initial training
----------------
The model is first trained with the
:ref:`Basic Optimizer Training Algorithm <basic_optimizer>` to establish a
reasonable baseline before the rule-growing procedure begins.

.. code-block:: python

    trainer = nft.Basic_optimizer_training_algorithm(
        epochs=5000,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW,
        optimizer_params={'lr': 1e-3, 'weight_decay': 1e-2},
        early_stopping=nft.EarlyStopping(patience=60)
    )

    trainer(model, train_loader, val_loader)

Initial evaluation
------------------
.. code-block:: python

    pred = model.predict(x_test)

    acc        = accuracy_score(y_test, pred)
    prec       = precision_score(y_test, pred, average='weighted', zero_division=0)
    recall     = recall_score(y_test, pred, average='weighted', zero_division=0)
    f1         = f1_score(y_test, pred, average='weighted', zero_division=0)
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

    Accuracy: 0.5846153846153846
    Precision: 0.6485067873303167
    Recall: 0.5846153846153846
    f1 score: 0.5797903356799868 

    Confusion Matrix:
    [[ 8  6  6  0  1  0]
     [ 2 19  1  1  0  0]
     [ 0  4  1  0  0  0]
     [ 0  3  0  1  0  0]
     [ 0  0  0  0  2  1]
     [ 0  2  0  0  0  7]] 

    Classification Report:
                  precision    recall  f1-score   support

               0       0.80      0.38      0.52        21
               1       0.56      0.83      0.67        23
               2       0.12      0.20      0.15         5
               3       0.50      0.25      0.33         4
               4       0.67      0.67      0.67         3
               5       0.88      0.78      0.82         9

        accuracy                           0.58        65
       macro avg       0.59      0.52      0.53        65
    weighted avg       0.65      0.58      0.58        65

Custom strategy: greedy rule-growing
--------------------------------------
The greedy rule-growing procedure iteratively attempts to expand the rule
base. At each step, a new rule is added centered on a training sample from
the class with the lowest current recall. The new rule's parameters are
fine-tuned in isolation; if validation loss improves, the rule is retained
and a global readaptation step is performed over all parameters. Otherwise,
the rule is discarded. The procedure terminates when a maximum number of
consecutive failed attempts is reached.

Helper function
^^^^^^^^^^^^^^^
.. code-block:: python

    loss_function = nn.CrossEntropyLoss()

    def val_loss(model):
        with torch.no_grad():
            return sum(
                loss_function(model(xb), yb) for xb, yb in val_loader
            ) / len(val_loader)

Hyperparameters
^^^^^^^^^^^^^^^^
.. code-block:: python

    max_failed_attempts      = 5

    single_adaptation_lr     = 0.005
    single_adaptation_epochs = 500
    single_patience          = 30

    global_adaptation_lr     = 0.001
    global_adaptation_epochs = 1000
    global_patience          = 60

Rule-growing loop
^^^^^^^^^^^^^^^^^^
.. code-block:: python

    failed_attempts = 0
    best_loss = val_loss(model)
    print(f"Initial val loss: {best_loss:.4f} | Rules: {model.rules}")
    print("=" * 60)

    while failed_attempts < max_failed_attempts:

        # Identify the worst-recall class
        with torch.no_grad():
            pred_train = model.predict(x_train)
        recalls = recall_score(
            y_train.numpy(), pred_train.numpy(), average=None, zero_division=0
        )
        worst_class = int(recalls.argmin())
        print(f"Recalls per class: {[f'{r:.2f}' for r in recalls]}")
        print(f"Worst class: {worst_class} (recall={recalls[worst_class]:.2f})")

        # Add a rule centered on a sample from the worst class
        class_indices = (y_train == worst_class).nonzero(as_tuple=True)[0]
        idx   = class_indices[torch.randint(0, len(class_indices), (1,))]
        means = x_train[idx].to(torch.float32)
        stds  = torch.full_like(means, 0.25)
        model.add_rules(means, stds)
        print(f"Rule added. Total rules: {model.rules}")

        # Fine-tune only the new rule's parameters
        new_params = [
            model.get_premises_as_parameters_list()[-1],
            model.get_consequents_as_parameters_list()[-1]
        ]
        opt_new = torch.optim.AdamW(
            new_params, lr=single_adaptation_lr, weight_decay=0.01
        )
        best_single_loss  = val_loss(model)
        patience_counter  = 0

        for epoch in range(single_adaptation_epochs):
            for xb, yb in train_loader:
                opt_new.zero_grad()
                loss_function(model(xb), yb).backward()
                opt_new.step()
            current = val_loss(model)
            if current < best_single_loss:
                best_single_loss = current
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= single_patience:
                print(f"  Single adaptation stopped at epoch {epoch + 1}"
                      f" | val loss: {current:.4f}")
                break

        val_after_single = val_loss(model)
        print(f"Val loss after single adaptation: {val_after_single:.4f}"
              f" (before: {best_loss:.4f})")

        # Retain or discard the new rule
        if val_after_single < best_loss:
            print("Rule RETAINED. Running global readaptation...")
            opt_all = torch.optim.AdamW(
                model.parameters(), lr=global_adaptation_lr, weight_decay=0.01
            )
            best_global_loss = val_after_single
            patience_counter = 0

            for epoch in range(global_adaptation_epochs):
                for xb, yb in train_loader:
                    opt_all.zero_grad()
                    loss_function(model(xb), yb).backward()
                    opt_all.step()
                current = val_loss(model)
                if current < best_global_loss:
                    best_global_loss = current
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= global_patience:
                    print(f"  Global adaptation stopped at epoch {epoch + 1}"
                          f" | val loss: {current:.4f}")
                    break

            best_loss       = val_loss(model)
            failed_attempts = 0
            print(f"Val loss after global adaptation: {best_loss:.4f}")
        else:
            model.remove_rules([model.rules - 1])
            failed_attempts += 1
            print(f"Rule DISCARDED. Failed attempts:"
                  f" {failed_attempts}/{max_failed_attempts}")

        print(f"Rules: {model.rules} | Best val loss: {best_loss:.4f}")
        print("-" * 60)

    print(f"\nFinal number of rules: {model.rules}")

.. code-block:: text

    Initial val loss: 0.8744 | Rules: 5
    ============================================================
    Recalls per class: ['0.69', '0.79', '0.40', '1.00', '0.80', '0.81']
    Worst class: 2 (recall=0.40)
    Rule added. Total rules: 6
      Single adaptation stopped at epoch 33 | val loss: 0.8744
    Val loss after single adaptation: 0.8744 (before: 0.8744)
    Rule RETAINED. Running global readaptation...
      Global adaptation stopped at epoch 62 | val loss: 0.9219
    Val loss after global adaptation: 0.9219
    Rules: 6 | Best val loss: 0.9219
    ------------------------------------------------------------
    Recalls per class: ['0.74', '0.71', '0.70', '1.00', '1.00', '0.81']
    Worst class: 2 (recall=0.70)
    ...
    ...
    ...
    Rules: 7 | Best val loss: 0.9153
    ------------------------------------------------------------

    Final number of rules: 7

Final evaluation
----------------
.. code-block:: python

    pred = model.predict(x_test)

    acc        = accuracy_score(y_test, pred)
    prec       = precision_score(y_test, pred, average='weighted', zero_division=0)
    recall     = recall_score(y_test, pred, average='weighted', zero_division=0)
    f1         = f1_score(y_test, pred, average='weighted', zero_division=0)
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

    Accuracy: 0.6153846153846154
    Precision: 0.6494505494505495
    Recall: 0.6153846153846154
    f1 score: 0.622604365590791 

    Confusion Matrix:
    [[12  3  5  0  1  0]
     [ 3 17  2  1  0  0]
     [ 1  3  1  0  0  0]
     [ 0  3  0  1  0  0]
     [ 0  0  0  0  2  1]
     [ 0  2  0  0  0  7]] 

    Classification Report:
                  precision    recall  f1-score   support

               0       0.75      0.57      0.65        21
               1       0.61      0.74      0.67        23
               2       0.12      0.20      0.15         5
               3       0.50      0.25      0.33         4
               4       0.67      0.67      0.67         3
               5       0.88      0.78      0.82         9

        accuracy                           0.62        65
       macro avg       0.59      0.53      0.55        65
    weighted avg       0.65      0.62      0.62        65

.. note::
    The built-in :ref:`SONFIS <sonfis-usage>` algorithm provides a
    self-organizing alternative to this custom procedure, encapsulating
    rule growing, splitting, and pruning within a single training loop
    operating directly on ``rule_reduced_ANFIS`` models.