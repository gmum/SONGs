import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


class Linear4D(nn.Module):
    __constants__ = ['num_models', 'in_features', 'out_features', 'num_nodes']
    num_models: int
    num_nodes: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, num_models: int, num_nodes: int, in_features: int, out_features: int, bias: bool = True,
                 beta: bool = False) -> None:
        super(Linear4D, self).__init__()
        self.num_models = num_models
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor(num_models, in_features, out_features, num_nodes))
        if bias:
            self.bias = Parameter(Tensor(num_models, out_features, num_nodes))
        else:
            self.register_parameter('bias', None)
        if beta:
            self.beta = Parameter(Tensor(num_models, out_features, num_nodes))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.beta is not None:
            nn.init.constant_(self.beta, 1)

    def forward(self, input_x: Tensor) -> Tensor:
        assert input_x.dim() >= 2, 'Input must be at least 2D'
        if input_x.dim() == 2:
            output = torch.einsum('gion,bi->gbon', self.weight, input_x)
        else:
            output = torch.einsum('gion,gbin->gbon', self.weight, input_x)
        if self.bias is not None:
            output = output + self.bias.unsqueeze(1)  # gbon x gon = gbon
        if self.beta is not None:
            output = output * self.beta.unsqueeze(1)  # gbon x gon = gbon
        return output

    def extra_repr(self) -> str:
        return f'num_models={self.num_models}, num_nodes={self.num_nodes}, in_features={self.in_features}, ' \
               f'out_features={self.out_features}{"" if self.bias is None else f", bias"}' \
               f'{"" if self.beta is None else f", beta"}'

    def l2_loss(self) -> Tensor:
        l2 = torch.einsum('mion,mion->mn', self.weight, self.weight).mean(-1)
        if self.bias is not None:
            l2 += torch.einsum('mon,mon->mn', self.bias, self.bias).mean(-1)
        if self.beta is not None:
            l2 += torch.einsum('mon,mon->mn', self.beta, self.beta).mean(-1)
        return l2
