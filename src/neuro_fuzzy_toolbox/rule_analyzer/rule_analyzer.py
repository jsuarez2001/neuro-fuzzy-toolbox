import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

from neuro_fuzzy_toolbox.models import ANFIS, h_ANFIS, rule_reduced_ANFIS

class RulesAnalyzer:
    """
    Analysis and interpretability tool for ANFIS rules.

    Provides methods for:

    - Identifying the most active rules for specific samples.
    - Exporting rules in human-readable IF-THEN format.
    - Analyzing rule contributions to model predictions.
    - Visualizing and estimating rule similarity.
    """
    
    def __init__(self, model):
        """
        Initializes the RulesAnalyzer.

        Args:
            model (ANFIS | h_ANFIS | rule_reduced_ANFIS): Trained ANFIS model
                to analyze.
        """
        self.model = model
        self.linguistic_labels = {}
        
        self.num_outputs = self.model.outputs
        self.output_type = self.model._output_type
        self.is_classification = (self.output_type == 'softmax')
        
        
    def _standardize_input(self, x):
        """
        Ensures that the input tensor has the correct shape for the model.

        Args:
            x (torch.Tensor): Input sample.

        Returns:
            torch.Tensor: Input sample with standardized shape ``(1, n_features)``.

        Raises:
            ValueError: If ``x`` is not a 1D tensor or a 2D tensor with a single sample.
        """
        if x.dim() == 1:
            return x.unsqueeze(0)  # Convertir a (1, features)
        elif x.dim() == 2 and x.shape[0] == 1:
            return x  # Ya tiene forma (1, features)
        else:
            raise ValueError(f"Input x must be a 1D tensor or a 2D tensor with a single sample. Current shape: {x.shape}")
        
    
    def layers_outputs(self, x):
        """
        Returns the relevant intermediate layer outputs of the model for a
        single input sample.
        
        Note:
            If the model has ``output_type='default'``, the output dictionary
            includes: ``'membership values'``, ``'firing levels'``,
            ``'norm firing levels'``, ``'consequent outputs'``,
            ``'rules contribution'``, and ``'final output'``. If the model
            has ``output_type='softmax'``, the dictionary additionally includes
            ``'logits'`` (the output before the softmax function).
        
        Args:
            x (torch.Tensor): Input sample of shape ``(n_features,)`` or ``(1, n_features)``.
        
        Returns:
            dict: Dictionary containing the relevant outputs of each layer.
        """
        x = self._standardize_input(x)
        
        with torch.no_grad():
            membership_values = self.model._fuzzification_layer(x)
            firing_levels = self.model._firing_levels_layer(membership_values)
            norm_firing_levels = self.model._normalization_layer(firing_levels)
            weighted_rules_outputs = self.model._consequent_layer(x, norm_firing_levels)
            output = self.model._output_layer(weighted_rules_outputs, return_probs=False)
            
        dict_output = {
            'membership values': membership_values,
            'firing levels': firing_levels,
            'norm firing levels': norm_firing_levels,
            'consequent outputs': self.model.get_all_consequents_outputs(x, weighted=False),
            'rules contribution': weighted_rules_outputs,
        }
        
        if self.is_classification:
            dict_output['logits'] = output
            with torch.no_grad():
                dict_output['final output'] = self.model._output_layer(weighted_rules_outputs, return_probs=True)
            
        else:
            dict_output['final output'] = output

        return dict_output
    
    
    def _classification_rule_scores(self, logits, probs, rules_contributions, class_idx):
        """
        Computes post-hoc rule relevance measures for a specific class.

        Args:
            logits (torch.Tensor): Model logits for the analyzed sample, of shape ``(n_classes,)``.
            probs (torch.Tensor): Final model probabilities for the analyzed sample, of shape ``(n_classes,)``.
            rules_contributions (torch.Tensor): Contribution of each rule to each class in the logit space, of shape ``(n_classes, n_rules)``.
            class_idx (int): Index of the target class :math:`c`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors of shape ``(n_rules,)`` containing
            ``I_logit_margin_max``, ``I_logit_margin_mean``, and ``I_prob``, respectively.

        Note:
            Let :math:`\\Delta z_{r,c}` be the contribution of rule :math:`r` to the logit of class :math:`c`, and let
            :math:`\\Delta \\mathbf{z}_r` be the full vector of contributions of that rule across all classes.

            The computed measures are:

            **1) Maximum logit margin per class**

            .. math::

                I^{(c)}_{\\text{logit\\_margin\\_max}}(r)
                =
                \\Delta z_{r,c}
                -
                \\max_{j \\neq c} \\Delta z_{r,j}

            Favors rules that increase the logit of the target class more than any other class.

            **2) Mean logit margin per class**

            .. math::

                I^{(c)}_{\\text{logit\\_margin\\_mean}}(r)
                =
                \\Delta z_{r,c}
                -
                \\frac{1}{C-1} \\sum_{j \\neq c} \\Delta z_{r,j}

            Compares the rule's contribution to the target class against the average of its contributions to all other classes.

            **3) Leave-one-rule-out probability**

            .. math::

                I^{(c)}_{\\text{prob}}(r)
                =
                p_c(\\mathbf{z})
                -
                p_c(\\mathbf{z} - \\Delta \\mathbf{z}_r)

            where :math:`p_c(\\mathbf{z})` is the softmax probability of the target class using all rules, and
            :math:`p_c(\\mathbf{z} - \\Delta \\mathbf{z}_r)` is the probability obtained by removing the contribution of rule
            :math:`r`.
            
            A positive value indicates that the rule supports the probability of the target class; a negative value indicates that
            it harms it.
        """
        no_pred_classes_contributions = torch.cat([rules_contributions[:class_idx], rules_contributions[class_idx+1:]], dim=0)
        pred_class_contribution = rules_contributions[class_idx, :]
        
        I_logit_margin_max = pred_class_contribution - torch.max(no_pred_classes_contributions, dim=0).values
        
        I_logit_margin_mean = pred_class_contribution - torch.mean(no_pred_classes_contributions, dim=0)
        
        I_prob = (probs - nn.functional.softmax(logits - rules_contributions.t(), dim=1))[:, class_idx] # leave one rule out -> probs
        """ this is the same as:
            I_prob = []
            for i in range(model4.rules):
                z_without_r = logits - contribution[:,i]
                p_without_r = nn.functional.softmax(z_without_r, dim=0)
                I_prob_r = real_prob[pred_idx] - p_without_r[pred_idx]
                I_prob.append(I_prob_r)
            I_prob = torch.tensor(I_prob)
        """

        return I_logit_margin_max, I_logit_margin_mean, I_prob
    
    
    def top_activated_rules(self, x, top_k=None, output_idx=None, sort_by='firing_levels'):
        """
        Identifies and ranks the top-k most active rules for a given input
        sample.

        Args:
            x (torch.Tensor): Input sample of shape ``(n_features,)`` or ``(1, n_features)``.
            top_k (int, optional): Number of top rules to return. If ``None``, all rules are included. Defaults to ``None``.
            output_idx (int, optional): Index of the output to analyze. Only used in regression models with multiple outputs;
                ignored otherwise. Defaults to ``None``.
            sort_by (str, optional): Criterion for ranking the active rules. Available options are ``'firing_levels'``,
                ``'abs_rules_outputs'``, ``'rules_outputs'``, ``'abs_contribution'``, and ``'contribution'``. 
                For models with ``output_type='softmax'``, the additional options ``'leave_one_rule_out'``, ``'logit_margin'``,
                and ``'logit_margin_mean'`` are also available. Defaults to ``'firing_levels'``.

        Returns:
            pandas.DataFrame | dict: The return type depends on the model's output type and the value of ``output_idx``:

            - **Regression models** (``output_type='default'``): Returns a ``pandas.DataFrame`` with columns ``'rule_id'``,
              ``'firing_level'``, ``'rule_output'``, and ``'contribution'`` when a single output is analyzed (either because 
              the model has one output or ``output_idx`` is specified). Returns a ``dict[str, pandas.DataFrame]`` with keys
              ``'output_0'``, ``'output_1'``, ... when the model has multiple outputs and ``output_idx`` is ``None``.

            - **Classification models** (``output_type='softmax'``): Returns a ``pandas.DataFrame`` with columns ``'rule_id'``,
              ``'firing_level'``, ``'rule_output'``, ``'contribution'``, ``'I_logit_margin_max'``, ``'I_logit_margin_mean'``, and
              ``'I_prob'`` when ``output_idx`` is specified. Returns a ``dict[str, pandas.DataFrame]`` with keys ``'class_0'``,
              ``'class_1'``, ... (or custom class label keys if the model uses custom class ids) when ``output_idx`` is ``None``.

        Note:
            Available sorting criteria:

            - ``'firing_levels'``: Sorts by normalized firing levels (:math:`w`).
            - ``'abs_rules_outputs'``: Sorts by the absolute value of each rule's individual output before weighting by firing levels (:math:`f(x)` without multiplying by :math:`w`).
            - ``'rules_outputs'``: Sorts by each rule's individual output before weighting by firing levels.
            - ``'abs_contribution'``: Sorts by the absolute value of each rule's contribution to the final output (:math:`f(x) \\cdot w`).
            - ``'contribution'``: Sorts by each rule's contribution to the final output.
            - ``'leave_one_rule_out'`` (``output_type='softmax'`` only): Sorts by the impact on the target class probability when the rule is removed, computed as the difference between the full-model probability and the leave-one-out probability.
            - ``'logit_margin'`` (``output_type='softmax'`` only): Sorts by the difference between the rule's contribution to the target class logit and its contribution to the highest-scoring alternative class.
            - ``'logit_margin_mean'`` (``output_type='softmax'`` only): Sorts by the difference between the rule's contribution to the target class logit and the mean of its contributions to all other classes.
        """
        x = self._standardize_input(x)
        
        all_layers_outputs = self.layers_outputs(x)
        
        # firing levels
        w = all_layers_outputs['norm firing levels'].squeeze(0)  # (R,)
        
        # consequent raw outputs (unweighted): (O, B, R) -> (O, R)
        consequent_outputs = all_layers_outputs['consequent outputs'][:, 0, :]
        
        # each rule output per model output/class: (O, R)
        rules_contributions = all_layers_outputs['rules contribution'].squeeze(1)
        
        # decide outputs/classes to analyze
        if output_idx is not None:
            outputs_to_analyze = [output_idx]
        elif self.num_outputs == 1:
            outputs_to_analyze = [0]
        else:
            outputs_to_analyze = list(range(self.num_outputs))
        
        results_dict = {}
        
        ##################################################################
        ######################### CLASSIFICATION #########################
        ##################################################################
        
        if self.is_classification:
            logits = all_layers_outputs['logits'].squeeze(0)  # (C,)
            pred_probs = all_layers_outputs['final output'].squeeze(0)  # (C,)
            
            for out_idx in outputs_to_analyze:
                I_logit_margin_max, I_logit_margin_mean, I_prob = self._classification_rule_scores(
                    logits=logits,
                    probs=pred_probs,
                    rules_contributions=rules_contributions,
                    class_idx=out_idx
                )
            
                if sort_by == "firing_levels":
                    sorted_indices = torch.argsort(w, descending=True)
                elif sort_by == "abs_rules_outputs":
                    sorted_indices = torch.argsort(torch.abs(consequent_outputs[out_idx, :]), descending=True)
                elif sort_by == "rules_outputs":
                    sorted_indices = torch.argsort(consequent_outputs[out_idx, :], descending=True)
                elif sort_by == "abs_contribution":
                    sorted_indices = torch.argsort(torch.abs(rules_contributions[out_idx, :]), descending=True)
                elif sort_by == "contribution":
                    sorted_indices = torch.argsort(rules_contributions[out_idx, :], descending=True)
                elif sort_by == "leave_one_rule_out":
                    sorted_indices = torch.argsort(I_prob, descending=True)
                elif sort_by == "logit_margin":
                    sorted_indices = torch.argsort(I_logit_margin_max, descending=True)
                elif sort_by == "logit_margin_mean":
                    sorted_indices = torch.argsort(I_logit_margin_mean, descending=True)
                else:
                    raise ValueError(
                        f"sort_by='{sort_by}' is not a valid option. Use 'firing_levels', 'abs_rules_outputs', 'rules_outputs', 'abs_contribution', 'contribution', 'leave_one_rule_out', 'logit_margin', or 'logit_margin_mean'."
                    )
                
                if top_k == None:
                    top_k_indices = sorted_indices[:self.model.rules]
                else:
                    top_k_indices = sorted_indices[:top_k]
                
                rows = []
                for idx in top_k_indices:
                    rid = idx.item()
                    rows.append({
                        "rule_id": rid + 1,
                        "firing_level": w[idx].item(),
                        "rule_output": consequent_outputs[out_idx, idx].item(),
                        "contribution": rules_contributions[out_idx, idx].item(),
                        "I_logit_margin_max": I_logit_margin_max[idx].item(),
                        "I_logit_margin_mean": I_logit_margin_mean[idx].item(),
                        "I_prob": I_prob[idx].item(),
                    })
            
                if self.model._custom_classes:
                    results_dict[f"class_{self.model._classes[out_idx].item()}"] = pd.DataFrame(rows)
                else:
                    results_dict[f"class_{out_idx}"] = pd.DataFrame(rows)
                
            return results_dict[f"class_{self.model._classes[outputs_to_analyze[0]].item()}"] if len(outputs_to_analyze) == 1 else results_dict 
        
        ##################################################################
        ########################### REGRESSION ###########################
        ##################################################################
        
        else:
        
            for out_idx in outputs_to_analyze:
                # sorting
                if sort_by == "firing_levels":
                    sorted_indices = torch.argsort(w, descending=True)
                elif sort_by == "abs_rules_outputs":
                    sorted_indices = torch.argsort(torch.abs(consequent_outputs[out_idx, :]), descending=True)
                elif sort_by == "rules_outputs":
                    sorted_indices = torch.argsort(consequent_outputs[out_idx, :], descending=True)
                elif sort_by == "abs_contribution":
                    sorted_indices = torch.argsort(torch.abs(rules_contributions[out_idx, :]), descending=True)
                elif sort_by == "contribution":
                    sorted_indices = torch.argsort(rules_contributions[out_idx, :], descending=True)
                else:
                    raise ValueError(
                        f"sort_by='{sort_by}' is not a valid option. Use 'firing_levels', 'abs_rules_outputs', 'rules_outputs', 'abs_contribution', or 'contribution'."
                    )

                if top_k == None:
                    top_k_indices = sorted_indices[:self.model.rules]
                else:
                    top_k_indices = sorted_indices[:top_k]

                rows = []
                for idx in top_k_indices:
                    rid = idx.item()
                    rows.append({
                        "rule_id": rid + 1,
                        "firing_level": w[idx].item(),
                        "rule_output": consequent_outputs[out_idx, idx].item(),
                        "contribution": rules_contributions[out_idx, idx].item(),
                    })

                results_dict[f"output_{out_idx}"] = pd.DataFrame(rows)

            return results_dict[f"output_{outputs_to_analyze[0]}"] if len(outputs_to_analyze) == 1 else results_dict
    
    
    def explain_prediction(self, x, top_k=None, alpha_cut=0.85, sort_by="firing_levels", show=[], output_idx=None):
        """
        Generates a textual explanation of the model's prediction for a given sample, based on the most active rules and their contributions.

        Args:
            x (torch.Tensor): Input sample to analyze.
            top_k (int, optional): Number of top rules to include. If ``None``, all rules are included. Defaults to ``None``.
            alpha_cut (float, optional): Minimum membership value used to define the membership intervals for each rule's antecedent. Defaults to ``0.85``.
            sort_by (str, optional): Criterion for ranking the active rules. Available options are ``'firing_levels'``, ``'abs_rules_outputs'``, ``'rules_outputs'``, ``'abs_contribution'``, and ``'contribution'``. For models with ``output_type='softmax'``, the additional options ``'leave_one_rule_out'``, ``'logit_margin'``, and ``'logit_margin_mean'`` are also available. Defaults to ``'firing_levels'``.
            show (list[str], optional): List of additional metrics to display alongside each rule in classification models. Available options are ``'leave_one_rule_out'``, ``'logit_margin'``, and ``'logit_margin_mean'``. Defaults to ``[]``.
            output_idx (int, optional): Index of the output to analyze (0-indexed). Defaults to ``None``.

        Returns:
            str: Textual explanation of the prediction based on the most active rules and their contributions.

        Note:
            Regarding ``output_idx``:

            - In regression models (``output_type='default'``), it is only used when the model has multiple outputs, to indicate which output the explanation refers to. It is ignored for single-output regression models.
            - In classification models (``output_type='softmax'``), it is ignored entirely, as the explanation always focuses on the predicted class.

            For available sorting criteria, see :meth:`top_activated_rules`.
        """
        x = self._standardize_input(x)
        
        with torch.no_grad():
            if self.is_classification:
                with torch.no_grad():
                    probs = self.model(x, return_probs=True)[0]  # (C,)
                    logits = self.model(x, return_probs=False)[0]  # (C,)
                pred_idx = torch.argmax(probs).item()
                pred = self.model.predict(x)
                
                explanation = "=" * 70 + "\n"
                explanation += "PREDICTION EXPLANATION\n"
                explanation += "=" * 70 + "\n\n"
                
                explanation += f"Predicted class: {pred.item()}\n"
                explanation += f"Predicted probability: {probs[pred_idx].item():.4f}\n\n"
                
                explanation += "Logits and probabilities:\n"
                for i in range(self.num_outputs):
                    cname = f"Class {self.model._classes[i]}"
                    explanation += f"  {cname}: logit={logits[i].item():.4f}, p={probs[i].item():.4f}\n"
                explanation += "\n"
                
                output_idx = pred_idx
                explanation += f"Explaining predicted class: {pred.item()}\n\n"
                
                top_rules = self.top_activated_rules(x, top_k, None, sort_by=sort_by)
                
                explanation += f"Top rules (sorted by {self._get_sort_type_str(sort_by)}):\n"
                explanation += "-" * 70 + "\n\n"
                
                key = f"class_{self.model._classes[output_idx].item()}" if self.model._custom_classes else f"class_{output_idx}"

                for _, row in top_rules[key].iterrows():
                    rule_id = row['rule_id'].astype(np.int64)
                    firing_level = row['firing_level']
                    rule_output = row['rule_output']
                    contribution = row['contribution']
                    I_logit_margin_max = row['I_logit_margin_max']
                    I_logit_margin_mean = row['I_logit_margin_mean']
                    I_prob = row['I_prob']
                    
                    explanation += f"Rule {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}"
                    if sort_by == 'leave_one_rule_out' or 'leave_one_rule_out' in show:
                        explanation += f" | I_prob={I_prob:+.4f}"
                    if sort_by == 'logit_margin' or 'logit_margin' in show:
                        explanation += f" | I_logit_margin_max={I_logit_margin_max:+.4f}"
                    if sort_by == 'logit_margin_mean' or 'logit_margin_mean' in show:
                        explanation += f" | I_logit_margin_mean={I_logit_margin_mean:+.4f}"
                    explanation += "\n"
                    
                    rule_desc = self._get_rule_description(alpha_cut, rule_id - 1, x)
                    explanation += f"  {rule_desc}\n\n"
                    
                return explanation
                
            else:
                pred = self.model.predict(x)
                
                explanation = "=" * 70 + "\n"
                if self.num_outputs == 1:
                    explanation += "PREDICTION EXPLANATION\n"
                    explanation += "=" * 70 + "\n\n"
                    explanation += f"Prediction: {pred.item():.4f}\n\n"
                else:
                    if output_idx is None:
                        output_idx = 0
                    pred = pred.squeeze()  # (O,)
                    explanation += "PREDICTION EXPLANATION (MULTIPLE OUTPUTS)\n"
                    explanation += "=" * 70 + "\n\n"
                    explanation += f"Explaining output {output_idx + 1}: {pred[output_idx].item():.4f}\n"
                    explanation += "\n"
                    
                top_rules = self.top_activated_rules(x, top_k, None, sort_by=sort_by)
                
                explanation += f"Top active rules (sorted by {self._get_sort_type_str(sort_by)}):\n"
                explanation += "-" * 70 + "\n\n"
                
                if self.num_outputs == 1:
                    for _, row in top_rules.iterrows():
                        rule_id = int(row['rule_id'])
                        firing_level = row['firing_level']
                        rule_output = row['rule_output']
                        contribution = row['contribution']

                        explanation += f"Rule {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}\n"
                        rule_desc = self._get_rule_description(alpha_cut, rule_id - 1, x)
                        explanation += f"  {rule_desc}\n\n"
                        
                else:
                    for _, row in top_rules[f"output_{output_idx}"].iterrows():
                        rule_id = int(row['rule_id'])
                        firing_level = row['firing_level']
                        rule_output = row['rule_output']
                        contribution = row['contribution']

                        explanation += f"Rule {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}\n"
                        rule_desc = self._get_rule_description(alpha_cut, rule_id - 1, x, output_idx)
                        explanation += f"  {rule_desc}\n\n"
                return explanation
            
    def show_fuzzy_sets(self, alpha_cut=0.85):
        """
        Generates a textual representation of the fuzzy sets (IF part) of all rules in the model.

        Args:
            alpha_cut (float, optional): Minimum membership value used to define the membership intervals. Defaults to ``0.85``.

        Returns:
            str: Text containing the antecedents of all rules in the model.

        Note:
            This method does not depend on a specific input sample. Its purpose is to display the global fuzzy structure of the model, showing the fuzzy sets associated with each rule based on their alpha-cuts.
        """
        output = "=" * 70 + "\n"
        output += "MODEL FUZZY SETS\n"
        output += "=" * 70 + "\n\n"
        output += f"Total rules: {self.model.rules}\n\n"

        for rule_idx in range(self.model.rules):
            if_clause = self._get_rule_if_clause(alpha_cut, rule_idx)
            output += f"Rule {rule_idx + 1}:\n"
            output += f"  {if_clause}\n\n"

        return output
            
    def show_top_fuzzy_sets(self, x, top_k=None, alpha_cut=0.85, sort_by="firing_levels", show=[], output_idx=None):
        """
        Generates a textual representation of the antecedents (IF part) of the
        most relevant rules for a given input sample.

        Args:
            x (torch.Tensor): Input sample to analyze.
            top_k (int, optional): Number of top rules to include. If ``None``, all rules are included. Defaults to ``None``.
            alpha_cut (float, optional): Minimum membership value used to define the membership intervals. Defaults to ``0.85``.
            sort_by (str, optional): Criterion for ranking the active rules. Available options are ``'firing_levels'``, ``'abs_rules_outputs'``, ``'rules_outputs'``, ``'abs_contribution'``, and ``'contribution'``. For models with ``output_type='softmax'``, the additional options ``'leave_one_rule_out'``, ``'logit_margin'``, and ``'logit_margin_mean'`` are also available. Defaults to ``'firing_levels'``.
            show (list[str], optional): List of additional metrics to display alongside each rule in classification models. Available options are ``'leave_one_rule_out'``, ``'logit_margin'``, and ``'logit_margin_mean'``. Defaults to ``[]``.
            output_idx (int, optional): Index of the output to analyze (0-indexed) in regression models with multiple outputs. If ``None``, the first output is analyzed. Ignored in classification models, where the explanation always focuses on the predicted class. Defaults to ``None``.

        Returns:
            str: Text containing the antecedents (IF part) of the most relevant rules.

        Note:
            This method reuses the ranking criterion of :meth:`top_activated_rules` and is intended as a compact version
            of :meth:`explain_prediction`, showing only the fuzzy sets activated in the rule antecedents.
        """
        x = self._standardize_input(x)

        with torch.no_grad():
            if self.is_classification:
                probs = self.model(x, return_probs=True)[0]   # (C,)
                logits = self.model(x, return_probs=False)[0] # (C,)
                pred_idx = torch.argmax(probs).item()
                pred = self.model.predict(x)

                explanation = "=" * 70 + "\n"
                explanation += "TOP FUZZY SETS\n"
                explanation += "=" * 70 + "\n\n"

                explanation += f"Predicted class: {pred.item()}\n"
                explanation += f"Predicted probability: {probs[pred_idx].item():.4f}\n\n"

                explanation += "Logits and probabilities:\n"
                for i in range(self.num_outputs):
                    cname = f"Class {self.model._classes[i]}"
                    explanation += f"  {cname}: logit={logits[i].item():.4f}, p={probs[i].item():.4f}\n"
                explanation += "\n"

                output_idx = pred_idx
                explanation += f"Showing antecedents for predicted class: {pred.item()}\n\n"

                top_rules = self.top_activated_rules(x, top_k, None, sort_by=sort_by)

                explanation += f"Fuzzy sets of the most activated rules (sorted by {self._get_sort_type_str(sort_by)}):\n"
                explanation += "-" * 70 + "\n\n"

                key = f"class_{self.model._classes[output_idx].item()}" if self.model._custom_classes else f"class_{output_idx}"

                for _, row in top_rules[key].iterrows():
                    rule_id = int(row['rule_id'])
                    firing_level = row['firing_level']
                    rule_output = row['rule_output']
                    contribution = row['contribution']
                    I_logit_margin_max = row['I_logit_margin_max']
                    I_logit_margin_mean = row['I_logit_margin_mean']
                    I_prob = row['I_prob']

                    explanation += f"Rule {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}"

                    if sort_by == 'leave_one_rule_out' or 'leave_one_rule_out' in show:
                        explanation += f" | I_prob={I_prob:+.4f}"
                    if sort_by == 'logit_margin' or 'logit_margin' in show:
                        explanation += f" | I_logit_margin_max={I_logit_margin_max:+.4f}"
                    if sort_by == 'logit_margin_mean' or 'logit_margin_mean' in show:
                        explanation += f" | I_logit_margin_mean={I_logit_margin_mean:+.4f}"
                    explanation += "\n"

                    if_clause = self._get_rule_if_clause(alpha_cut, rule_id - 1)
                    explanation += f"  {if_clause}\n\n"

                return explanation

            else:
                pred = self.model.predict(x)

                explanation = "=" * 70 + "\n"
                if self.num_outputs == 1:
                    explanation += "TOP FUZZY SETS\n"
                    explanation += "=" * 70 + "\n\n"
                    explanation += f"Prediction: {pred.item():.4f}\n\n"
                else:
                    if output_idx is None:
                        output_idx = 0
                    pred = pred.squeeze()
                    explanation += "TOP FUZZY SETS (MULTIPLE OUTPUTS)\n"
                    explanation += "=" * 70 + "\n\n"
                    explanation += f"Explaining output {output_idx + 1}: {pred[output_idx].item():.4f}\n\n"

                top_rules = self.top_activated_rules(x, top_k, None, sort_by=sort_by)

                explanation += f"Fuzzy sets of the most activated rules (sorted by {self._get_sort_type_str(sort_by)}):\n"
                explanation += "-" * 70 + "\n\n"

                if self.num_outputs == 1:
                    for _, row in top_rules.iterrows():
                        rule_id = int(row['rule_id'])
                        firing_level = row['firing_level']
                        rule_output = row['rule_output']
                        contribution = row['contribution']

                        explanation += f"Rule {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}\n"
                        if_clause = self._get_rule_if_clause(alpha_cut, rule_id - 1)
                        explanation += f"  {if_clause}\n\n"

                else:
                    for _, row in top_rules[f"output_{output_idx}"].iterrows():
                        rule_id = int(row['rule_id'])
                        firing_level = row['firing_level']
                        rule_output = row['rule_output']
                        contribution = row['contribution']

                        explanation += f"Rule {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}\n"
                        if_clause = self._get_rule_if_clause(alpha_cut, rule_id - 1)
                        explanation += f"  {if_clause}\n\n"

                return explanation
            
    def _get_rule_if_clause(self, alpha_cut, rule_idx):
        """
        Generates the IF clause (antecedent) of a single rule.

        Args:
            alpha_cut (float): Minimum membership value used to define the membership intervals.
            rule_idx (int): Index of the rule to describe (0-indexed).

        Returns:
            str: Antecedent of the rule in human-readable format.

        Raises:
            ValueError: If ``rule_idx`` is out of range.

        Note:
            The antecedent is constructed from the membership function parameters of each input feature and their alpha-cut intervals.
        """
        premises = self.model.get_premises()

        if rule_idx >= self.model.rules:
            raise ValueError(
                f"rule_idx={rule_idx} is out of range. The model has {self.model.rules} rules."
            )

        if_parts = []

        # ANFIS / h_ANFIS sin reducción de reglas
        if isinstance(self.model, ANFIS) or (isinstance(self.model, h_ANFIS) and not self.model._rule_reduced):
            if isinstance(self.model, ANFIS):
                mf_dist = self.model._fuzzification_layer._mf_distribution
            else:
                mf_dist = [self.model.num_mfs] * self.model._input_size

            mf_indices = []
            temp_idx = rule_idx
            for i in range(len(mf_dist) - 1, -1, -1):
                mf_indices.insert(0, temp_idx % mf_dist[i])
                temp_idx //= mf_dist[i]

            for input_idx, (premise, mf_idx) in enumerate(zip(premises, mf_indices)):
                params = []
                i = 0
                for _ in self.model._fuzzification_layer._membership_function._params:
                    params.append(premise[mf_idx, i].item())
                    i += 1

                range_np = self.model._fuzzification_layer._membership_function._simple_alpha_cut(alpha_cut, *params)
                range_str = f"∈ [{range_np[0].item():.2f}, {range_np[1].item():.2f}]"

                feature_name = self.model.features[input_idx]
                if_parts.append(f"{feature_name} {range_str}")

        # rule_reduced_ANFIS
        else:
            for input_idx, premise in enumerate(premises):
                params = []
                i = 0
                for _ in self.model._fuzzification_layer._membership_function._params:
                    params.append(premise[rule_idx, i].item())
                    i += 1

                range_np = self.model._fuzzification_layer._membership_function._simple_alpha_cut(alpha_cut, *params)
                range_str = f"∈ [{range_np[0].item():.2f}, {range_np[1].item():.2f}]"

                feature_name = self.model.features[input_idx]
                if_parts.append(f"{feature_name} {range_str}")

        if_clause = " AND ".join(if_parts) if if_parts else "N/A"
        return f"IF {if_clause}"
            
    def _get_rule_description(self, alpha_cut, rule_idx, x, output_idx=None):
        """
        Generates a full IF-THEN textual description of a specific rule.

        Note:
            The docstring of this method should be placed before the first line of code. Currently it appears after ``x = self._standardize_input(x)``.

        Args:
            alpha_cut (float): Minimum membership value used to define the membership intervals.
            rule_idx (int): Index of the rule to describe (0-indexed).
            x (torch.Tensor): Input sample for which the rule is described.
            output_idx (int, optional): Index of the output to analyze (0-indexed) in regression models with multiple outputs. If ``None``, the first output is analyzed. Ignored in classification models and single-output regression models. Defaults to ``None``.

        Returns:
            str: Full IF-THEN textual description of the rule.

        Raises:
            ValueError: If ``rule_idx`` is out of range.
        """
        x = self._standardize_input(x)
        
        premises = self.model.get_premises()
        consequents = self.model.get_consequents()

        if rule_idx >= self.model.rules:
            raise ValueError(f"rule_idx={rule_idx} is out of range. The model has {self.model.rules} rules.")
        
        ##################################################################
        ############################### IF ###############################
        ##################################################################
        if_parts = []
        
        # ANFIS
        if isinstance(self.model, ANFIS) or (isinstance(self.model, h_ANFIS) and not self.model._rule_reduced):
            # Necesitamos mapear el índice de regla a las combinaciones de MFs
            if isinstance(self.model, ANFIS):
                mf_dist = self.model._fuzzification_layer._mf_distribution
            else:
                mf_dist = [self.model.num_mfs]*self.model._input_size
                
            # Se hace un proceso de división sucesiva para mapear el índice de regla a los índices de MFs correspondientes a cada input
            mf_indices = []
            temp_idx = rule_idx
            for i in range(len(mf_dist) - 1, -1, -1):
                mf_indices.insert(0, temp_idx % mf_dist[i])
                temp_idx //= mf_dist[i]
                
            for input_idx, (premise, mf_idx) in enumerate(zip(premises, mf_indices)):
                params = []
                i = 0
                for _ in self.model._fuzzification_layer._membership_function._params:
                    params.append(premise[mf_idx, i].item())
                    i += 1
                
                range_np = self.model._fuzzification_layer._membership_function._simple_alpha_cut(alpha_cut, *params)
                range_str = f"∈ [{range_np[0].item():.2f}, {range_np[1].item():.2f}]"
                
                feature_name = self.model.features[input_idx]
                if_parts.append(f"{feature_name} {range_str}")
                    
        # rule_reduced_ANFIS
        else:
            for input_idx, premise in enumerate(premises):
                params = []
                i = 0
                for _ in self.model._fuzzification_layer._membership_function._params:
                    params.append(premise[rule_idx, i].item())
                    i += 1
                
                range_np = self.model._fuzzification_layer._membership_function._simple_alpha_cut(alpha_cut, *params)
                range_str = f"∈ [{range_np[0].item():.2f}, {range_np[1].item():.2f}]"
                
                feature_name = self.model.features[input_idx]
                if_parts.append(f"{feature_name} {range_str}")
                    
        if_clause = " AND ".join(if_parts) if if_parts else "N/A"
        
        ##################################################################
        ############################## THEN ##############################
        ##################################################################
        
        then_parts = {}
        
        for temp_output_idx in range(self.num_outputs):
            coefs = consequents[temp_output_idx, rule_idx, :-1]
            bias = consequents[temp_output_idx, rule_idx, -1]
            
            terms = []
            for i, coef in enumerate(coefs):
                if abs(coef.item()) > 1e-6:
                    terms.append(f"{coef.item():.3f}*{self.model.features[i]}")
            terms.append(f"{bias.item():.3f}")
            then_expr = " + ".join(terms).replace("+ -", "- ")
            
            if self.is_classification:
                cname = self.model._classes[temp_output_idx]
                out_name = f"f_{cname}(x)"
                then_parts[temp_output_idx] = [out_name, then_expr]
            else:
                out_name = f"output_{temp_output_idx + 1}"
                then_parts[temp_output_idx] = [out_name, then_expr]
                
        ##################################################################
        ############################## RULE ##############################
        ##################################################################
        
        rule = ""
        if_part_len = len(if_clause) + 11  # "  IF " + if_clause + " THEN "
        if self.is_classification:
            rule = f"IF {if_clause} THEN "
            for i in range(self.num_outputs):
                rule += f"{then_parts[i][0]} = {then_parts[i][1]} \n"
                rule += " " * if_part_len
        else:
            if output_idx is None:
                output_idx = 0
            rule = f"IF {if_clause} THEN f(x) = {then_parts[output_idx][1]}"
            
        return rule
    
    def _get_sort_type_str(self, sort_by):
        """
        Returns a human-readable description of a sorting criterion given its code.

        Args:
            sort_by (str): Sorting criterion code. Available options are ``'firing_levels'``, ``'abs_rules_outputs'``, ``'rules_outputs'``, ``'abs_contribution'``, and ``'contribution'``. For classification models, the additional options ``'leave_one_rule_out'``, ``'logit_margin'``, and ``'logit_margin_mean'`` are also available.

        Returns:
            str: Human-readable description of the sorting criterion.
        """
        sort_types = {
            'firing_levels': "firing levels",
            'abs_rules_outputs': "absolute values of unweighted rule outputs",
            'rules_outputs': "unweighted rule outputs",
        }
        
        if self.is_classification:
            sort_types['abs_contribution'] = "absolute contribution to the class logit"
            sort_types['contribution'] = "contribution to the class logit"
            sort_types['leave_one_rule_out'] = "change in predicted class probability when the rule is removed"
            sort_types['logit_margin'] = "logit margin (predicted class vs. highest-scoring alternative)"
            sort_types['logit_margin_mean'] = "logit margin (predicted class vs. mean of other classes)"
        else:
            sort_types['abs_contribution'] = "absolute contribution to the final output"
            sort_types['contribution'] = "contribution to the final output"
            
        return sort_types[sort_by]