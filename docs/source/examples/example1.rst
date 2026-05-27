.. _example1:

Example 1: Multiclass Classification on Iris Dataset
======================================================

This example demonstrates the standard toolbox workflow on the
`Iris dataset <https://doi.org/10.24432/C56C76>`_, a four-feature,
three-class classification benchmark. An ``h_ANFIS`` model is trained
using the :ref:`Basic Optimizer Training Algorithm <basic_optimizer>` with
early stopping, and the trained model is analyzed using
:ref:`RulesAnalyzer <Rule_Analyzer-class>`.

Imports and reproducibility
----------------------------
.. code-block:: python

    from sklearn.datasets import load_iris
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
The four features are scaled to [0, 1] using ``MinMaxScaler``. The dataset
is split into training (70%), validation (16%), and test (14%) sets using
stratified sampling to preserve class proportions.

.. code-block:: python

    iris = load_iris()
    X, y = iris.data, iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    scaler = MinMaxScaler(feature_range=(0, 1))

    x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
    x_val   = torch.tensor(scaler.transform(x_val),       dtype=torch.float32)
    x_test  = torch.tensor(scaler.transform(x_test),      dtype=torch.float32)

    y_train = torch.tensor(y_train)
    y_val   = torch.tensor(y_val)
    y_test  = torch.tensor(y_test)

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
An ``h_ANFIS`` model is instantiated with 3 MFs per input feature and a
softmax output layer for three-class classification. Premise parameters are
initialized from the training data distribution, and consequent parameters
are estimated by regularized least squares prior to gradient-based training.

.. code-block:: python

    model = nft.h_ANFIS(
        input_size=4,
        num_mfs=3,
        outputs=3,
        output_type='softmax',
        features=['sepal length', 'sepal width', 'petal length', 'petal width']
    )

    model.init_premises(x_train)
    model.init_consequents(x_train, y_train, ridge_lambda=0.1)

Learning algorithm
------------------
The model is trained with ``AdamW`` and early stopping monitoring the
validation loss.

.. code-block:: python

    trainer = nft.Basic_optimizer_training_algorithm(
        epochs=500,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW,
        optimizer_params={'lr': 1e-3, 'weight_decay': 1e-2},
        early_stopping=nft.EarlyStopping(patience=30, delta=1e-4)
    )

    trainer(model, train_loader, val_loader)

Evaluation
----------
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

    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    f1 score: 1.0 
    
    Confusion Matrix:
    [[15  0  0]
     [ 0 15  0]
     [ 0  0 15]] 
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        15
               1       1.00      1.00      1.00        15
               2       1.00      1.00      1.00        15
    
        accuracy                           1.00        45
       macro avg       1.00      1.00      1.00        45
    weighted avg       1.00      1.00      1.00        45

Rule structure analysis
-----------------------
Once the model is trained, the rule base can be inspected in tabular form
and the learned MFs can be visualized per input feature.

.. code-block:: python

    print(model.get_rules_structure().to_string())

    model.plot_premises(group_by_dim=True)

The ``RulesAnalyzer`` class provides rule-level contribution analysis for a
specific input sample. The example below retrieves the top 3 rules ranked by
their leave-one-rule-out impact on the predicted class probability:

.. code-block:: python

    analyzer = nft.RulesAnalyzer(model)

    top_rules = analyzer.top_activated_rules(
        x_test[0:1], top_k=3, sort_by='leave_one_rule_out'
    )

    for class_label, df in top_rules.items():
        print(f"{class_label}:")
        print(df.to_string(), "\n")

.. code-block:: text

    class_0:
       rule_id  firing_level  rule_output  contribution  I_logit_margin_max  I_logit_margin_mean  I_prob
    0       53  1.751464e-13    -0.144441 -2.529828e-14       -4.459777e-14        -2.978208e-14     0.0
    1       62  2.913474e-21    -0.137805 -4.014923e-22       -1.052639e-21        -6.202652e-22     0.0
    2       61  9.106845e-24    -0.114750 -1.045014e-24       -2.155477e-24        -1.098971e-24     0.0 

    class_1:
       rule_id  firing_level  rule_output  contribution  I_logit_margin_max  I_logit_margin_mean        I_prob
    0       15  8.753379e-05     0.658258  5.761984e-05            0.000027             0.000172  1.317821e-07
    1       17  1.669576e-06     1.864878  3.113556e-06            0.000005             0.000007  2.235174e-08
    2       41  2.438206e-07     2.951781  7.197052e-07            0.000001             0.000001  4.656613e-09 

    class_2:
       rule_id  firing_level  rule_output  contribution  I_logit_margin_max  I_logit_margin_mean    I_prob
    0       45      0.972177     3.168142      3.079994            5.205152             5.228931  0.623935
    1       18      0.024140     2.764272      0.066728            0.128098             0.133860  0.001298
    2       42      0.003525     2.528276      0.008913            0.013312             0.013376  0.000122

.. code-block:: python

    explanation = analyzer.explain_prediction(x_test[0:1], top_k=3, sort_by="leave_one_rule_out")
    print(explanation)

.. code-block:: text
    
    ======================================================================
    PREDICTION EXPLANATION
    ======================================================================

    Predicted class: 2
    Predicted probability: 0.9908

    Logits and probabilities:
      Class 0: logit=-2.2506, p=0.0044
      Class 1: logit=-2.1909, p=0.0047
      Class 2: logit=3.1558, p=0.9908

    Explaining predicted class: 2

    Top rules (sorted by change in predicted class probability when the rule is removed):
    ----------------------------------------------------------------------

    Rule 45 | w=0.9722 | f(x)=3.1681 | contrib=+3.0800 | I_prob=+0.6239
      IF sepal length ∈ [0.57, 0.80] AND sepal width ∈ [0.30, 0.79] AND petal length ∈ [0.80, 1.17] AND petal width ∈ [0.84, 1.16] THEN f_0(x) = -0.591*sepal length - 0.548*sepal width - 0.578*petal length - 0.561*petal width - 0.577 
                                                                                                                                        f_1(x) = -0.596*sepal length - 0.541*sepal width - 0.585*petal length - 0.533*petal width - 0.553 
                                                                                                                                        f_2(x) = 0.822*sepal length + 0.640*sepal width + 0.854*petal length + 0.769*petal width + 0.907 


    Rule 18 | w=0.0241 | f(x)=2.7643 | contrib=+0.0667 | I_prob=+0.0013
      IF sepal length ∈ [-0.14, 0.32] AND sepal width ∈ [0.30, 0.79] AND petal length ∈ [0.80, 1.17] AND petal width ∈ [0.84, 1.16] THEN f_0(x) = -0.800*sepal length - 0.808*sepal width - 0.767*petal length - 0.744*petal width - 0.765 
                                                                                                                                         f_1(x) = -0.676*sepal length - 0.588*sepal width - 0.663*petal length - 0.662*petal width - 0.647 
                                                                                                                                         f_2(x) = 0.734*sepal length + 0.686*sepal width + 0.712*petal length + 0.701*petal width + 0.702 


    Rule 42 | w=0.0035 | f(x)=2.5283 | contrib=+0.0089 | I_prob=+0.0001
      IF sepal length ∈ [0.57, 0.80] AND sepal width ∈ [0.30, 0.79] AND petal length ∈ [0.32, 0.58] AND petal width ∈ [0.84, 1.16] THEN f_0(x) = -0.335*sepal length - 0.316*sepal width - 0.331*petal length - 0.318*petal width - 0.340 
                                                                                                                                        f_1(x) = -0.430*sepal length - 0.037*sepal width - 0.390*petal length - 0.167*petal width - 0.518 
                                                                                                                                        f_2(x) = 0.707*sepal length + 0.202*sepal width + 0.711*petal length + 0.513*petal width + 0.953

.. note::
    For a complete description of the analysis methods available in
    ``RulesAnalyzer``, see :ref:`Rule Inspection and Analysis <Rule_Inspection_and_Analysis>`.