.. _Rule_Inspection_and_Analysis:

Rule Inspection and Analysis
============================

Neuro-Fuzzy Toolbox provides a ``RulesAnalyzer`` class for inspecting and
analyzing trained ANFIS models. It offers tools for identifying the most
active rules for a given input, ranking them by different relevance criteria,
generating human-readable IF-THEN rule descriptions, and inspecting the
intermediate outputs of each layer.

``RulesAnalyzer`` is compatible with all model variants available in
Neuro-Fuzzy Toolbox (``ANFIS``, ``h_ANFIS``, and ``rule_reduced_ANFIS``).

.. note::
    For more details on the implementation of this module, see :ref:`Rules Analyzer <Rule_Analyzer>`.

1. Instantiation
----------------
``RulesAnalyzer`` takes a trained ANFIS model as its only argument:

.. code-block:: python

    import neuro_fuzzy_toolbox as nft

    analyzer = nft.RulesAnalyzer(model)

All methods described in this section operate on the model passed at
instantiation. No additional setup is required.

2. Inspecting intermediate layer outputs
-----------------------------------------
The ``layers_outputs(x)`` method returns the outputs of the relevant
intermediate layers of the model for a single input sample, as a dictionary
of tensors.

.. code-block:: python

    outputs = analyzer.layers_outputs(x)
    print(list(outputs.keys()))

For a **regression** model (``output_type='default'``), the dictionary
contains the following keys:

.. code-block:: text

    ['membership values', 'firing levels', 'norm firing levels',
     'consequent outputs', 'rules contribution', 'final output']

For a **classification** model (``output_type='softmax'``), the key
``'logits'`` is also included, and ``'final output'`` contains the
softmax probabilities:

.. code-block:: text

    ['membership values', 'firing levels', 'norm firing levels',
     'consequent outputs', 'rules contribution', 'logits', 'final output']

Each value is a tensor whose shape reflects the batch dimension (always 1
for a single sample), the number of rules, and the number of outputs or
classes. For example, for a classification model with 4 input features, 3
MFs per feature, and 3 output classes:

.. code-block:: python

    for k, v in outputs.items():
        print(f"{k}: shape={v.shape}")

.. code-block:: text

    membership values:  shape=torch.Size([1, 4, 3])
    firing levels:      shape=torch.Size([1, 81])
    norm firing levels: shape=torch.Size([1, 81])
    consequent outputs: shape=torch.Size([3, 1, 81])
    rules contribution: shape=torch.Size([3, 1, 81])
    logits:             shape=torch.Size([1, 3])
    final output:       shape=torch.Size([1, 3])

.. tip::
    This method is useful for debugging or for gaining a deeper understanding
    of how the model processes a specific sample. The intermediate outputs can
    also be used to build custom analysis pipelines on top of the model.

3. Identifying the most active rules
--------------------------------------
The ``top_activated_rules(x, top_k, output_idx, sort_by)`` method identifies
and ranks the rules of the model for a given input sample.

Return type
^^^^^^^^^^^
The return type depends on the model and the value of ``output_idx``:

.. table::
    :align: center

    +--------------------+---------------------------+---------------------------------------+
    | Model type         | ``output_idx``            | Return type                           |
    +====================+===========================+=======================================+
    | Any, single output | ``None`` or specified     | ``pandas.DataFrame``                  |
    +--------------------+---------------------------+---------------------------------------+
    | Regression,        | ``None``                  | ``dict[str, DataFrame]``              |
    | multiple outputs   |                           | (keys: ``'output_0'``, ``'output_1'``)|
    +--------------------+---------------------------+---------------------------------------+
    | Regression,        | specified                 | ``pandas.DataFrame``                  |
    | multiple outputs   |                           |                                       |
    +--------------------+---------------------------+---------------------------------------+
    | Classification     | ``None``                  | ``dict[str, DataFrame]``              |
    |                    |                           | (keys: ``'class_0'``, ``'class_1'``…) |
    +--------------------+---------------------------+---------------------------------------+
    | Classification     | specified                 | ``pandas.DataFrame``                  |
    +--------------------+---------------------------+---------------------------------------+

DataFrame columns
^^^^^^^^^^^^^^^^^
For **regression** models, each DataFrame contains the following columns:

.. code-block:: text

    rule_id | firing_level | rule_output | contribution

For **classification** models, three additional relevance measures are
included:

.. code-block:: text

    rule_id | firing_level | rule_output | contribution |
    I_logit_margin_max | I_logit_margin_mean | I_prob

where:

- ``firing_level``: normalized firing level (:math:`\bar{w}`) of the rule.
- ``rule_output``: unweighted consequent output of the rule (:math:`f_r(x)`).
- ``contribution``: weighted rule output (:math:`\bar{w}_r \cdot f_r(x)`), i.e., the rule's contribution to the final model output.
- ``I_logit_margin_max``: difference between the rule's contribution to the target class logit and its contribution to the highest-scoring alternative class.
- ``I_logit_margin_mean``: difference between the rule's contribution to the target class logit and the mean of its contributions to all other classes.
- ``I_prob``: change in the predicted class probability when the rule is removed (leave-one-rule-out).

Sorting criteria
^^^^^^^^^^^^^^^^
The ``sort_by`` parameter controls how rules are ranked. The following
options are available for all model types:

- ``'firing_levels'``: sorted by normalized firing level (default).
- ``'abs_rules_outputs'``: sorted by the absolute value of the unweighted rule output.
- ``'rules_outputs'``: sorted by the unweighted rule output.
- ``'abs_contribution'``: sorted by the absolute value of the rule contribution.
- ``'contribution'``: sorted by the rule contribution.

The following options are additionally available for classification models:

- ``'leave_one_rule_out'``: sorted by the change in predicted class probability when the rule is removed.
- ``'logit_margin'``: sorted by ``I_logit_margin_max``.
- ``'logit_margin_mean'``: sorted by ``I_logit_margin_mean``.

Examples
^^^^^^^^
Retrieving all rules for all classes of a classification model (returns a
``dict``):

.. code-block:: python

    all_rules = analyzer.top_activated_rules(x)
    # Returns: {'class_0': DataFrame, 'class_1': DataFrame, 'class_2': DataFrame}

    for class_key, df in all_rules.items():
        print(f"{class_key}:")
        print(df.to_string())

Retrieving the top 3 rules for a specific class, sorted by
``leave_one_rule_out``:

.. code-block:: python

    top3 = analyzer.top_activated_rules(
        x, top_k=3, output_idx=2, sort_by='leave_one_rule_out'
    )
    # output_idx specified -> returns a DataFrame directly
    print(top3.to_string())

.. code-block:: text

       rule_id  firing_level  rule_output  contribution  I_logit_margin_max  I_logit_margin_mean  I_prob
    0       45  9.721768e-01     3.168060  3.079969e+00        5.252710e+00         2.719001e+00    0.6239
    1       18  2.413955e-02     2.764296  6.673256e-02        1.448516e-01         6.704671e-02    0.0013
    2       42  3.525269e-03     1.789678  6.310424e-03        1.430988e-02         6.921119e-03    0.0001

For a single-output regression model (always returns a ``DataFrame``):

.. code-block:: python

    reg_top3 = analyzer.top_activated_rules(
        x, top_k=3, sort_by='abs_contribution'
    )
    print(reg_top3.to_string())

.. code-block:: text

       rule_id  firing_level  rule_output  contribution
    0       16      0.542703    -0.877229     -0.476074
    1       13      0.393028    -0.789003     -0.310100
    2        4      0.009217    -1.756315     -0.016188

4. Full prediction explanation
--------------------------------
The ``explain_prediction(x, top_k, alpha_cut, sort_by, show, output_idx)``
method generates a textual explanation of the model's prediction for a given
sample. It combines rule statistics with human-readable IF-THEN descriptions
of the rule antecedents, derived from the alpha-cuts of the membership
functions.

For **classification** models, the explanation focuses on the predicted class
and always includes the logits and probabilities for all classes. The
``output_idx`` parameter is ignored since the predicted class is used
automatically:

.. code-block:: python

    print(analyzer.explain_prediction(x, top_k=3, sort_by='leave_one_rule_out',
                                      show=['logit_margin']))

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

    Rule 45 | w=0.9722 | f(x)=3.1681 | contrib=+3.0800 | I_prob=+0.6239 | I_logit_margin_max=+5.2052
      IF sepal length (cm) ∈ [0.57, 0.80] AND sepal width (cm) ∈ [0.30, 0.79]
         AND petal length (cm) ∈ [0.80, 1.17] AND petal width (cm) ∈ [0.84, 1.16]
      THEN f_0(x) = ... f_1(x) = ... f_2(x) = ...

    ...

For **regression** models, the explanation includes the prediction value and
the IF-THEN expression for each rule:

.. code-block:: python

    print(analyzer.explain_prediction(x, top_k=3))

.. code-block:: text

    ======================================================================
    PREDICTION EXPLANATION
    ======================================================================

    Prediction: -0.8301

    Top active rules (sorted by firing levels):
    ----------------------------------------------------------------------

    Rule 16 | w=0.5427 | f(x)=-0.8772 | contrib=-0.4761
      IF x0 ∈ [-0.53, 0.48] AND x1 ∈ [0.57, 1.40] AND x2 ∈ [-1.40, -0.55]
      THEN f(x) = 0.902*x0 + 0.003*x1 + 1.034*x2 - 0.021

    Rule 13 | w=0.3930 | f(x)=-0.7890 | contrib=-0.3101
      IF x0 ∈ [-0.53, 0.48] AND x1 ∈ [-0.45, 0.44] AND x2 ∈ [-1.40, -0.55]
      THEN f(x) = -0.002*x0 + 0.023*x1 + 0.945*x2 - 0.023

    Rule 17 | w=0.0156 | f(x)=-0.8081 | contrib=-0.0126
      IF x0 ∈ [-0.53, 0.48] AND x1 ∈ [0.57, 1.40] AND x2 ∈ [-0.42, 0.38]
      THEN f(x) = 0.922*x0 - 0.009*x1 + 0.987*x2 + 0.016

For **multi-output** regression models, specify ``output_idx`` to select
which output to explain:

.. code-block:: python

    print(analyzer.explain_prediction(x, top_k=3, output_idx=0))

.. code-block:: text

    ======================================================================
    PREDICTION EXPLANATION (MULTIPLE OUTPUTS)
    ======================================================================

    Explaining output 1: -0.8300
    ...

.. note::
    The ``alpha_cut`` parameter (default ``0.85``) controls the membership
    threshold used to derive the input intervals in the IF clauses. A higher
    value produces narrower intervals that correspond more precisely to the
    core of each fuzzy set.

5. Inspecting the global fuzzy structure
-----------------------------------------
The ``show_fuzzy_sets(alpha_cut)`` method generates a textual listing of the
antecedents (IF part) of all rules in the model. Unlike
``explain_prediction``, it does not depend on a specific input sample — it
reflects the global structure of the model's rule base:

.. code-block:: python

    print(analyzer.show_fuzzy_sets(alpha_cut=0.85))

.. code-block:: text

    ======================================================================
    MODEL FUZZY SETS
    ======================================================================

    Total rules: 27

    Rule 1:
      IF x0 ∈ [-1.41, -0.53] AND x1 ∈ [-1.41, -0.41] AND x2 ∈ [-1.40, -0.55]

    Rule 2:
      IF x0 ∈ [-1.41, -0.53] AND x1 ∈ [-1.41, -0.41] AND x2 ∈ [-0.42, 0.38]

    ...

6. Sample-specific antecedents
--------------------------------
The ``show_top_fuzzy_sets(x, top_k, alpha_cut, sort_by, show, output_idx)``
method is a compact version of ``explain_prediction`` that shows only the IF
part of the most relevant rules for a given sample, without the THEN
expressions. It uses the same sorting and ranking logic as
``top_activated_rules``:

.. code-block:: python

    print(analyzer.show_top_fuzzy_sets(x, top_k=3,
                                       sort_by='leave_one_rule_out'))

For a **classification** model:

.. code-block:: text

    ======================================================================
    TOP FUZZY SETS
    ======================================================================

    Predicted class: 2
    Predicted probability: 0.9908

    Logits and probabilities:
      Class 0: logit=-2.2506, p=0.0044
      Class 1: logit=-2.1909, p=0.0047
      Class 2: logit=3.1558, p=0.9908

    Showing antecedents for predicted class: 2

    Fuzzy sets of the most activated rules (sorted by change in predicted class probability when the rule is removed):
    ----------------------------------------------------------------------

    Rule 45 | w=0.9722 | f(x)=3.1681 | contrib=+3.0800 | I_prob=+0.6239
      IF sepal length (cm) ∈ [0.57, 0.80] AND sepal width (cm) ∈ [0.30, 0.79]
         AND petal length (cm) ∈ [0.80, 1.17] AND petal width (cm) ∈ [0.84, 1.16]

    ...

For a **regression** model:

.. code-block:: text

    ======================================================================
    TOP FUZZY SETS
    ======================================================================

    Prediction: -0.8301

    Fuzzy sets of the most activated rules (sorted by firing levels):
    ----------------------------------------------------------------------

    Rule 16 | w=0.5427 | f(x)=-0.8772 | contrib=-0.4761
      IF x0 ∈ [-0.53, 0.48] AND x1 ∈ [0.57, 1.40] AND x2 ∈ [-1.40, -0.55]

    ...

For **multi-output** regression models, use ``output_idx`` to specify which
output the ranking should refer to:

.. code-block:: python

    print(analyzer.show_top_fuzzy_sets(x, top_k=3, output_idx=0))

.. code-block:: text

    ======================================================================
    TOP FUZZY SETS (MULTIPLE OUTPUTS)
    ======================================================================

    Explaining output 1: -0.8300
    ...