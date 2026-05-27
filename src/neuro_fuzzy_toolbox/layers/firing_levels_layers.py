import torch
import torch.nn as nn


class FiringLevelsLayer(nn.Module):
    """
    Firing levels layer for a general Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Computes the firing level of each fuzzy rule by applying the T-norm (product) operator across the membership degrees
    of each input feature. Designed to handle a general ANFIS model where different input features may have different numbers
    of MFs.
    """
    def __init__(self, mf_distribution):
        """
        Initializes a new FiringLevelsLayer instance.

        Args:
            mf_distribution (list[int]): Number of MFs for each input feature.
        """
        super(FiringLevelsLayer, self).__init__()
        self._firing_level_mask = (torch.arange(mf_distribution.max()).unsqueeze(1) < mf_distribution).t()
        self._rules = mf_distribution.prod()

    def forward(self, membership_values):
        """
        Forward pass of the firing levels layer.

        Computes the firing level of each rule as the product of the membership degrees of the corresponding MFs across all input features.

        Args:
            membership_values (torch.Tensor): Membership degrees of shape ``(batch_size, input_size, max_num_mfs)``,
                where ``max_num_mfs`` is the maximum number of MFs across all input features.

        Returns:
            torch.Tensor: Firing levels of shape ``(batch_size, num_mfs_1 * num_mfs_2 * ... * num_mfs_n)``,
            where the number of rules is the product of the number of MFs across all input features.
        """
        return torch.cat([torch.cartesian_prod(*[dim_mvs[dim_mask] for dim_mvs, dim_mask in zip(mvs, self._firing_level_mask)]).prod(dim=-1) for mvs in membership_values]).reshape(-1, self._rules)



class h_FiringLevelsLayer(nn.Module):
    """
    Firing levels layer for a homogeneous Adaptive Neuro-Fuzzy Inference System (ANFIS).
    
    Computes the firing level of each fuzzy rule by applying the T-norm (product) operator across the membership degrees 
    of each input feature. Assumes the same number of MFs for every input feature.
    
    Supports an optional rule-reduced mode that avoids the full combinatorial expansion of membership degrees. In this mode,
    only the membership degrees at matching indices across features are multiplied together, yielding a number of rules equal
    to the number of MFs per feature rather than ``num_mfs ** input_size``. For further details, see :ref:`rule-reduced ANFIS <rule-reduced ANFIS>`.
    """
    def __init__(self, rule_reduced=False):
        """
        Initializes a new h_FiringLevelsLayer instance.

        Args:
            rule_reduced (bool): If ``True``, uses rule-reduced firing level
                computation instead of the full combinatorial expansion.
                Defaults to ``False``.
        """
        super(h_FiringLevelsLayer, self).__init__()
        if rule_reduced:
            self._get_firing_levels = lambda membership_values: membership_values.prod(dim=membership_values.dim()-2)
        else:
            self._get_firing_levels = lambda membership_values: torch.cat([torch.cartesian_prod(*torch.unbind(t, dim=0)).prod(dim=-1) for t in membership_values]).reshape(-1, membership_values.shape[-1]**membership_values.shape[-2])

    def forward(self, membership_values):
        """
        Forward pass of the homogeneous firing levels layer.

        Computes the firing level of each rule as the product of the membership
        degrees of the corresponding MFs across all input features.

        Args:
            membership_values (torch.Tensor): Membership degrees of shape ``(batch_size, input_size, num_mfs)``.

        Returns:
            torch.Tensor: Firing levels of shape ``(batch_size, num_mfs ** input_size)`` in standard mode, or
            ``(batch_size, num_mfs)`` in rule-reduced mode.
        """
        return self._get_firing_levels(membership_values)
    
    
    
class rule_reduced_FiringLevelsLayer(nn.Module):
    """
    Firing levels layer for a rule-reduced Adaptive Neuro-Fuzzy Inference System (ANFIS).
    
    Computes the firing level of each fuzzy rule using the rule-reduced approach,
    where only the membership degrees at matching indices across features are
    multiplied together. This avoids the full combinatorial expansion, yielding
    a number of rules equal to the number of MFs per feature. Assumes the same
    number of MFs for every input feature.
    
    Optionally supports a default rule that adds an extra firing level to capture
    input combinations not covered by the reduced rule set.
    
    Warning:
        The default rule functionality is experimental and not fully supported.
    """
    def __init__(self, default_rule=False):
        """
        Initializes a new rule_reduced_FiringLevelsLayer instance.

        Args:
            default_rule (bool): If ``True``, appends an extra firing level representing a default rule to capture input combinations not
                covered by the reduced rule set. Defaults to ``False``.
                
        Warning:
            The ``default_rule`` option is experimental and not fully supported.
        """
        super(rule_reduced_FiringLevelsLayer, self).__init__()
        if default_rule:
            self._get_firing_levels = lambda firing_levels, input_size: torch.cat((firing_levels, torch.pow(1 - firing_levels.max(dim=1).values.unsqueeze(1), input_size)), dim=1)
        else:
            self._get_firing_levels = lambda firing_levels, input_size: firing_levels
            
    def forward(self, membership_values):
        """
        Forward pass of the rule-reduced firing levels layer.

        Computes the firing level of each rule by multiplying the membership degrees at matching indices across all input features.

        Args:
            membership_values (torch.Tensor): Membership degrees of shape ``(batch_size, input_size, num_mfs)``.

        Returns:
            torch.Tensor: Firing levels of shape ``(batch_size, num_mfs)``, or ``(batch_size, num_mfs + 1)`` if the default rule is enabled.
        """
        firing_levels = membership_values.prod(dim=membership_values.dim()-2)
        return self._get_firing_levels(firing_levels, membership_values.shape[1])



class NormalizationLayer(nn.Module):
    """
    Normalization layer for an Adaptive Neuro-Fuzzy Inference System (ANFIS).

    Normalizes the firing levels across all rules so that they sum to one,
    producing the normalized firing levels used by the consequent layer.

    Optionally supports a default rule that adds an extra firing level to capture
    input combinations not covered by the reduced rule set.
    
    Warning:
        The default rule functionality is experimental and not fully supported.
    """
    def __init__(self, default_rule=False):
        """
        Initializes a new NormalizationLayer instance.
        
        Args:
            default_rule (bool): If ``True``, the layer expects the last firing level in the input to correspond to the default rule. 
                In this case, all firing levels (including the default rule's) are summed for normalization, but the default rule's firing 
                level is excluded from the normalized output returned by :meth:`forward`.
                If ``False``, all firing levels are normalized and returned as-is.
                Defaults to ``False``.
        
        Warning:
            The default rule functionality is experimental and not fully supported.
        """
        super(NormalizationLayer, self).__init__()
        if default_rule:
            self._get_firing_levels = lambda firing_levels: firing_levels[:,:-1]
        else:
            self._get_firing_levels = lambda firing_levels: firing_levels

    def forward(self, w):
        """
        Forward pass of the normalization layer.

        Normalizes the firing levels so that they sum to one across all rules. If a zero sum is encountered, it is 
        replaced with one to avoid division by zero.

        Args:
            w (torch.Tensor): Firing levels of shape ``(batch_size, num_rules)``.

        Returns:
            torch.Tensor: Normalized firing levels of shape ``(batch_size, num_rules)``, or ``(batch_size, num_rules - 1)`` 
            if the default rule is enabled.
        """
        total = torch.sum(w, dim=-1, keepdim=True)
        total[total == 0] += 1
        norm_firing_levels = self._get_firing_levels(w)/total
        return norm_firing_levels