import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

from neuro_fuzzy_toolbox.models import ANFIS, h_ANFIS, rule_reduced_ANFIS

class RulesAnalyzer:
    """
    Clase para análisis e interpretabilidad de reglas ANFIS.
    
    Proporciona métodos para:  
    - Identificar reglas más activas para muestras específicas  
    - Exportar reglas en formato texto legible (IF-THEN)  
    - Asignar variables lingüisticas a intervalos personalizados  
    - Analizar contribuciones de reglas a predicciones  
    - Visualizar y estimar similitud de reglas
    """
    
    def __init__(self, model):
        """
        Inicializa el analizador de reglas.
        
        Args:
            model (ANFIS | h_ANFIS | rule_reduced_ANFIS): Modelo ANFIS entrenado.
        """
        self.model = model
        self.linguistic_labels = {}
        
        self.num_outputs = self.model.outputs
        self.output_type = self.model._output_type
        self.is_classification = (self.output_type == 'softmax')
        
        
    def _standardize_input(self, x):
        """
        Asegura que la entrada x tenga la forma adecuada para el modelo (1D o 2D).
        
        Args:
            x (torch.tensor): Muestra de entrada.
        
        Returns:
            torch.tensor: Muestra de entrada con forma estandarizada.
        """
        if x.dim() == 1:
            return x.unsqueeze(0)  # Convertir a (1, features)
        elif x.dim() == 2 and x.shape[0] == 1:
            return x  # Ya tiene forma (1, features)
        else:
            raise ValueError(f"Entrada x debe ser un tensor 1D o un tensor 2D con una sola muestra. Forma actual: {x.shape}")
        
    
    def layers_outputs(self, x):
        """
        Obtiene las salidas relevantes de cada capa del modelo para una muestra dada en un diccionario.
        
        Note:
            Si el modelo tiene tipo de salida 'default', el diccionario de salida incluye: 'membership values', 'firing levels', 'norm firing levels', 'consequent outputs', 'rules contributions' y 'final output'.  
            Si el modelo tiene tipo de salida 'softmax', se incluye además 'logits' (salida antes de softmax).
        
        Args:
            x (torch.tensor): Muestra de entrada (1D o 2D).
        
        Returns:
            dict: Diccionario con salidas relevantes de cada capa.
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
    
    
    def _logits_margin(self, pred_class_idx, logits):
        """
        Calcula el margen de logits para la clase predicha, definido como la diferencia entre el logit de la clase predicha y el logit de la clase con la segunda mayor probabilidad.
        
        Args:
            pred_class_idx (int): Índice de la clase predicha.
            logits (torch.tensor): Tensor con los logits de todas las clases (C,).
            
        Returns:
            float: Margen de logits para la clase predicha.
            
        Note:
            Un margen de logits más alto indica una mayor confianza en la predicción, ya que la clase predicha tiene un logit significativamente mayor que las demás clases. Este margen puede ser útil para
            evaluar la confianza de la predicción y para analizar cómo las reglas contribuyen a aumentar o disminuir este margen.
            
        """
    
    
    def top_activated_rules(self, x, top_k=None, output_idx=None, sort_by='firing_levels'):
        """
        Identifica las k reglas más activas para una muestra de entrada.

        Args:
            x (torch.tensor): Tensor con el dato de entrada a analizar.
            top_k (int, optional): Número de reglas top a retornar. Si no se indica, se incluyen todas las reglas. (Default: None)
            output_idx (int, optional): Índice de la salida a analizar. Si no se indica, se analiza todas las salidas. (Default: None)
            sort_by (str, optional): Criterio de ordenamiento de reglas activas. Puede ser 'firing_levels', 'abs_rules_outputs', 'rules_outputs', 'abs_contribution' o 'contribution'. Si el modelo tiene tipo salida 'softmax', se puede usar adicionalmente 'leave_one_rule_out', 'logit_margin' o 'logit_margin_mean'. (Default: 'firing_levels')
            
        Returns:
            - Si el modelo tiene tipo salida 'default':
                - Si el modelo tiene una sola salida o se especifica output_idx: dataframe con las reglas ordenadas por el criterio indicado, con columnas 'rule', 'firing_level', 'rule_output' y 'contribution'.
                - Si el modelo tiene múltiples salidas y no se especifica output_idx: dict con claves 'output_0', 'output_1', ... para cada salida, y valores que son dataframes con las reglas ordenadas.
            - Si el modelo tiene tipo de salida 'softmax':
                - Si se especifica output_idx: dataframe con las reglas ordenadas por el criterio indicado, con columnas 'rule', 'firing_level', 'rule_output', 'contribution'.
                - Si output_idx es None: dict con claves 'class_0', 'class_1', ... para cada salida, y valores que son dataframes con las reglas ordenadas. Si el modelo tiene ids de clase personalizados se usan esos ids en lugar de 'class_0', 'class_1', ...
        """
        x = self._standardize_input(x)
        
        all_layers_outputs = self.layers_outputs(x)
        
        # firing levels
        w = all_layers_outputs['norm firing levels'].squeeze(0)  # (R,)
        
        # consequent raw outputs (unweighted): (O, B, R) -> (O, R)
        consequent_outputs = all_layers_outputs['consequent outputs'][:, 0, :]
        
        # each rule output per model output/class: (O, R)
        rules_contributions = all_layers_outputs['rules contribution'].squeeze(1)
        
        if self.is_classification:
            logits = all_layers_outputs['logits'].squeeze(0)  # (C,)
            pred_probs = all_layers_outputs['final output'].squeeze(0)  # (C,)
            pred_idx = torch.argmax(all_layers_outputs['final output']).item()
            
            no_pred_classes_contributions = torch.cat([rules_contributions[:pred_idx], rules_contributions[pred_idx+1:]], dim=0)
            pred_class_contribution = rules_contributions[pred_idx, :]
            
            I_logit_margin_max = pred_class_contribution - torch.max(no_pred_classes_contributions, dim=0).values
            
            I_logit_margin_mean = pred_class_contribution - torch.mean(no_pred_classes_contributions, dim=0)
            
            I_prob = (pred_probs - nn.functional.softmax(logits - rules_contributions.t(), dim=1))[:, pred_idx] # leave one rule out -> probs
            """ this is the same as:
            I_prob = []
            for i in range(model4.rules):
                z_without_r = logits - contribution[:,i]
                p_without_r = nn.functional.softmax(z_without_r, dim=0)
                I_prob_r = real_prob[pred_idx] - p_without_r[pred_idx]
                I_prob.append(I_prob_r)
            I_prob = torch.tensor(I_prob)
            """
        
        # decide outputs/classes to analyze
        if output_idx is not None:
            outputs_to_analyze = [output_idx]
        elif self.num_outputs == 1:
            outputs_to_analyze = [0]
        else:
            outputs_to_analyze = list(range(self.num_outputs))
        
        results_dict = {}
        
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
                if self.is_classification:
                    if sort_by == "leave_one_rule_out":
                        sorted_indices = torch.argsort(I_prob, descending=True)
                    elif sort_by == "logit_margin":
                        sorted_indices = torch.argsort(I_logit_margin_max, descending=True)
                    elif sort_by == "logit_margin_mean":
                        sorted_indices = torch.argsort(I_logit_margin_mean, descending=True)
                    else:
                        raise ValueError(f"Modo sort_by='{sort_by}' no disponible. Use 'firing_levels', 'abs_rules_outputs', 'rules_outputs', 'abs_contribution', 'contribution', 'leave_one_rule_out', 'logit_margin' o 'logit_margin_mean'.")
                else:
                    raise ValueError(f"Modo sort_by='{sort_by}' no disponible. Use 'firing_levels', 'abs_rules_outputs', 'rules_outputs', 'abs_contribution' o 'contribution'.")

            if top_k == None:
                top_k_indices = sorted_indices[:self.model.rules]
            else:
                top_k_indices = sorted_indices[:top_k]
            
            rows = []
            for idx in top_k_indices:
                rid = idx.item()
                if self.is_classification:
                    rows.append({
                        "rule_id": rid + 1,
                        "firing_level": w[idx].item(),
                        "rule_output": consequent_outputs[out_idx, idx].item(),
                        "contribution": rules_contributions[out_idx, idx].item(),
                        "I_logit_margin_max": I_logit_margin_max[idx].item(),
                        "I_logit_margin_mean": I_logit_margin_mean[idx].item(),
                        "I_prob": I_prob[idx].item(),
                    })
                else:
                    rows.append({
                        "rule_id": rid + 1,
                        "firing_level": w[idx].item(),
                        "rule_output": consequent_outputs[out_idx, idx].item(),
                        "contribution": rules_contributions[out_idx, idx].item(),
                    })
                
            # SOFTMAX OUTPUT TYPE
            if self.is_classification:
                
                if self.model._custom_classes:
                    results_dict[f"class_{self.model._classes[out_idx].item()}"] = pd.DataFrame(rows)
                else:
                    results_dict[f"class_{out_idx}"] = pd.DataFrame(rows)
                    
            # DEFAULT OUTPUT TYPE
            else:
                results_dict[f"output_{out_idx}"] = pd.DataFrame(rows)
                
        if self.is_classification:
            return results_dict[f"class_{self.model._classes[outputs_to_analyze[0]].item()}"] if len(outputs_to_analyze) == 1 else results_dict
        else:
            return results_dict[f"output_{outputs_to_analyze[0]}"] if len(outputs_to_analyze) == 1 else results_dict
    
    
    def explain_prediction(self, x, top_k=None, alpha_cut=0.85, sort_by="firing_levels", output_idx=None):
        """
        Genera una explicación textual de la predicción del modelo para una muestra dada, basada en las reglas más activas y sus contribuciones a la predicción.
        
        Args:
            x (torch.tensor): Tensor con el dato de entrada a analizar.
            top_k (int, optional): Número de reglas top a considerar. Si no se indica, se incluyen todas las reglas. (Default: None)
            alpha_cut (float, optional): Valor de membresía mínimo con el que se definen los intervalos de pertenencia.
            sort_by (str, optional): Criterio de ordenamiento de reglas activas. Puede ser 'firing_levels', 'abs_rules_outputs', 'rules_outputs', 'abs_contribution' o 'contribution'. Si el modelo tiene tipo salida 'softmax', se puede usar adicionalmente 'leave_one_rule_out', 'logit_margin' o 'logit_margin_mean'. (Default: 'firing_levels')
            output_idx (int, optional): Índice de la salida (0-indexed) a analizar en un problema de regresión con múltiples salidas. Si no se indica, se analiza la primera salida. (Default: None)
        
        Returns:
            str: Explicación textual de la predicción basada en las reglas más activas y sus contribuciones.
            
        Note:
            output_idx solo se utiliza para modelos de regresión con múltiples salidas, y se incluye en la descripción para indicar a qué salida corresponde la explicación. En modelos de clasificación o regresión con una sola salida, output_idx se ignora.
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
                explanation += "EXPLICACIÓN DE PREDICCIÓN\n"
                explanation += "=" * 70 + "\n\n"
                
                explanation += f"Clase Predicha: {pred.item()}\n"
                explanation += f"Prob. predicha: {probs[pred_idx].item():.4f}\n\n"
                
                explanation += "Logits y probabilidades:\n"
                for i in range(self.num_outputs):
                    cname = f"Class {self.model._classes[i]}"
                    explanation += f"  {cname}: logit={logits[i].item():.4f}, p={probs[i].item():.4f}\n"
                explanation += "\n"
                
                output_idx = pred_idx
                explanation += f"Explicando clase predicha: {pred}\n\n"
                
                top_rules = self.top_activated_rules(x, top_k, None, sort_by=sort_by)
                
                explanation += f"Reglas principales (por {self._get_sort_type_str(sort_by)}):\n"
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
                    
                    explanation += f"Regla {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}"
                    if sort_by == 'leave_one_rule_out':
                        explanation += f" | I_prob={I_prob:+.4f}"
                    elif sort_by == 'logit_margin':
                        explanation += f" | I_logit_margin_max={I_logit_margin_max:+.4f}"
                    elif sort_by == 'logit_margin_mean':
                        explanation += f" | I_logit_margin_mean={I_logit_margin_mean:+.4f}"
                    explanation += "\n"
                    
                    rule_desc = self._get_rule_description(alpha_cut, rule_id - 1, x)
                    explanation += f"  {rule_desc}\n\n"
                    
                return explanation
                
            else:
                pred = self.model.predict(x)
                
                explanation = "=" * 70 + "\n"
                if self.num_outputs == 1:
                    explanation += "EXPLICACIÓN DE PREDICCIÓN\n"
                    explanation += "=" * 70 + "\n\n"
                    explanation += f"Predicción: {pred.item():.4f}\n\n"
                else:
                    if output_idx is None:
                        output_idx = 0
                    pred = pred.squeeze()  # (O,)
                    explanation += "EXPLICACIÓN DE PREDICCIÓN (MÚLTIPLES OUTPUTS)\n"
                    explanation += "=" * 70 + "\n\n"
                    explanation += f"Explicando output {output_idx + 1}: {pred[output_idx].item():.4f}\n"
                    explanation += "\n"
                    
                top_rules = self.top_activated_rules(x, top_k, None, sort_by=sort_by)
                
                explanation += f"Reglas principales activas (por {self._get_sort_type_str(sort_by)}):\n"
                explanation += "-" * 70 + "\n\n"
                
                if self.num_outputs == 1:
                    for _, row in top_rules.iterrows():
                        rule_id = int(row['rule_id'])
                        firing_level = row['firing_level']
                        rule_output = row['rule_output']
                        contribution = row['contribution']

                        explanation += f"Regla {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}\n"
                        rule_desc = self._get_rule_description(alpha_cut, rule_id - 1, x)
                        explanation += f"  {rule_desc}\n\n"
                        
                else:
                    for _, row in top_rules[f"output_{output_idx}"].iterrows():
                        rule_id = int(row['rule_id'])
                        firing_level = row['firing_level']
                        rule_output = row['rule_output']
                        contribution = row['contribution']

                        explanation += f"Regla {rule_id} | w={firing_level:.4f} | f(x)={rule_output:.4f} | contrib={contribution:+.4f}\n"
                        rule_desc = self._get_rule_description(alpha_cut, rule_id - 1, x, output_idx)
                        explanation += f"  {rule_desc}\n\n"
                return explanation
            
    def _get_rule_description(self, alpha_cut, rule_idx, x, output_idx=None):
        x = self._standardize_input(x)
        
        """
        Genera una descripción textual de una regla específica.
        
        Args:
            alpha_cut (float): Valor de membresía mínimo con el que se definen los intervalos de pertenencia.
            rule_idx (int): Índice de la regla a describir (0-indexed).
            x (torch.tensor): Muestra de entrada para la cual se describe la regla.
            output_idx (int, optional): Índice de la salida (0-indexed) a analizar en un problema de regresión con múltiples salidas. Si no se indica, se analiza la primera salida. (Default: None)
        
        Returns:
            str: Descripción textual de la regla.
            
        Note:
            output_idx solo se utiliza para modelos de regresión con múltiples salidas, y se incluye en la descripción para indicar a qué salida corresponde la explicación. En modelos de clasificación o regresión con una sola salida, output_idx se ignora.
        """
        premises = self.model.get_premises()
        consequents = self.model.get_consequents()

        if rule_idx >= self.model.rules:
            raise ValueError(f"rule_idx={rule_idx} fuera de rango. El modelo tiene {self.model.rules} reglas.")
        
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
                out_name = f"logit({cname})"
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
        Retorna una descripción legible del criterio de ordenamiento dado su código.
        
        Args:
            sort_by (str): Código del criterio de ordenamiento. Puede ser 'firing_levels', 'abs_rules_outputs', 'rules_outputs', 'abs_contribution' o 'contribution'.
            
        Returns:
            str: Descripción legible del criterio de ordenamiento.
        """
        sort_types = {
            'firing_levels': "firing levels",
            'abs_rules_outputs': "valores absolutos de las salidas de regla sin ponderar",
            'rules_outputs': "valores de las salidas de regla sin ponderar",
        }
        
        if self.is_classification:
            sort_types['abs_contribution'] = "contribución absoluta al logit de la clase"
            sort_types['contribution'] = "contribución al logit de la clase"
            sort_types['leave_one_rule_out'] = "diferencia en la probabilidad de la clase predicha al dejar fuera la regla"
            sort_types['logit_margin'] = "diferencia en logits (clase predicha vs siguiente mejor clase)"
            sort_types['logit_margin_mean'] = "diferencia en logits (clase predicha vs promedio de otras clases)"
        else:
            sort_types['abs_contribution'] = "contribución absoluta a la salida final"
            sort_types['contribution'] = "contribución a la salida final"
            
        return sort_types[sort_by]
    
    
    def plot_parallel_rules(self, highlight_rules=None, figsize=(14,8)):
        """
        Visualiza las reglas mediante coordenadas paralelas.
        
        Cada regla se representa como una línea que atraviesa ejes paralelos (uno por cada input),
        mostrando los centros de sus funciones de membresía, facilitando identificar reglas con rangos
        solapados.

        Args:
            highlight_rules (list, optional): Lista de índices de reglas a resaltar en la visualización. (Default: None).
            figsize (tuple, optional): Tamaño de la figura. (Default: (14,8)).
        """
        premises = self.model.get_premises()
        num_rules = self.model.rules
        
        rule_centers = []
        
        # ANFIS
        if isinstance(self.model, ANFIS) or (isinstance(self.model, h_ANFIS) and not self.model._rule_reduced):
            # Necesitamos mapear el índice de regla a las combinaciones de MFs
            if isinstance(self.model, ANFIS):
                mf_dist = self.model._fuzzification_layer._mf_distribution
            else:
                mf_dist = [self.model.num_mfs]*self.model._input_size
                
            for rule_idx in range(num_rules):
                centers = []
                temp_idx = rule_idx
                for i in range(len(mf_dist) - 1, -1, -1):
                    mf_idx = temp_idx % mf_dist[i]
                    temp_idx //= mf_dist[i]
                    
                    if mf_idx < premises[i].shape[0]:
                        centers.insert(0, self.model._fuzzification_layer._membership_function._get_center(*premises[i][mf_idx].tolist()))
                    else:
                        centers.insert(0, 0.0)
                rule_centers.append(centers)
        
        # rule_reduced_ANFIS
        else:
            for rule_idx in range(num_rules):
                centers = []
                for premise in premises:
                    if rule_idx < premise.shape[0]:
                        centers.append(self.model._fuzzification_layer._membership_function._get_center(*premise[rule_idx].tolist()))
                    else:
                        centers.append(0.0)
                rule_centers.append(centers)
                
        # Crear DataFrame para pandas parallel_coordinates
        df_data = []
        for rule_idx, centers in enumerate(rule_centers):
            row = {'rule': f'R{rule_idx+1}'}
            for feat_idx, center in enumerate(centers):
                row[self.model.features[feat_idx]] = center
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        if highlight_rules:
            colors = ['red' if f'R{i}' in [f'R{r}' for r in highlight_rules] 
                     else 'lightblue' for i in range(1, len(df)+1)]
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
            
        # Plotear coordenadas paralelas
        pd.plotting.parallel_coordinates(df, 'rule', ax=ax, 
                                        color=colors, alpha=0.6)
        
        ax.set_ylabel('Centros de Funciones de Membresía', fontsize=12)
        ax.set_title('Visualización de Reglas en Coordenadas Paralelas', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        
        # Ajustar leyenda si hay muchas reglas
        if len(df) > 15:
            ax.legend().set_visible(False)
            ax.text(1.02, 0.5, f'{len(df)} reglas\ntotal', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='center')
            
        plt.tight_layout()
        
        plt.show()