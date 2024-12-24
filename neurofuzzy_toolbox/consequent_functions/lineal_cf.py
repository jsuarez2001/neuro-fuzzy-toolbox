import torch

from consequent_functions import ConsequentFunction

class Linear_CF(ConsequentFunction):

    def forward(self, x, consequents, weights):
        return (torch.bmm(x.unsqueeze(0).expand(consequents[:, :, :-1].size(0), -1, -1), torch.transpose(consequents[:, :, :-1], 1, 2)) + consequents[:, :, -1].unsqueeze(1)).mul(weights.unsqueeze(0))

    def initialize_consequents(self, outputs, consequents_rules, input_size, dtype):
        return 2 * torch.rand(outputs, consequents_rules, input_size + 1, dtype=dtype) - 1