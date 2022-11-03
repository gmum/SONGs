from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.gumbel import Gumbel
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from song.shallow.utils.mnist_classifier import Net
from song.utils.layers import Linear4D


class NodeNoise(nn.Module):
    def __init__(self, loc: float = 0, scale: float = 1):
        super().__init__()
        self.noise = Gumbel(loc, scale)

    def forward(self, input_x: Tensor) -> Tensor:
        if self.training:
            size = input_x.shape
            noise = self.noise.sample((size[0], 2, *size[1:])).to(input_x.device)
            noise = noise[:, 0, ...] - noise[:, 1, ...]
            input_x = input_x + noise
        return input_x


class Graphs(nn.Module):
    __constants__ = ['hidden_layers_nodes', 'num_nodes', 'num_leaves', 'num_jumps']
    hidden_layers_nodes: List[int]
    num_graphs: int
    num_nodes: int
    num_leaves: int
    num_jumps: Optional[int]
    learning_roots: bool
    tau: float

    nodes: nn.Module
    M_left: Tensor
    M_right: Tensor

    def __init__(self, hidden_layers_nodes: List[int], num_nodes: int, num_leaves: int, num_jumps: Optional[int],
                 learning_roots: bool = False, num_graphs: int = 1, tau: float = None, use_bias: bool = True,
                 use_beta: bool = False, gumbel_nodes: bool = False, tau_regularization: float = None,
                 binarization_threshold: float = 0) -> None:
        super(Graphs, self).__init__()

        assert 0 <= binarization_threshold < 1
        self.binarization_threshold = binarization_threshold
        self.num_graphs = num_graphs
        self.hidden_layers_nodes = hidden_layers_nodes
        self.num_nodes = num_nodes
        self.num_leaves = num_leaves
        self.num_jumps = num_jumps
        self.learning_roots = learning_roots
        self.tau = tau  # non-negative scalar temperature for Gumbel-Softmax if None use typical 'softmax'
        self.tau_regularization = tau_regularization

        self.M_left = Parameter(Tensor(num_graphs, num_nodes + num_leaves, num_nodes))
        self.M_right = Parameter(Tensor(num_graphs, num_nodes + num_leaves, num_nodes))

        self.nodes = nn.Sequential()
        for i in range(1, len(hidden_layers_nodes)):
            self.nodes.add_module(f'dense{i:d}',
                                  Linear4D(num_graphs, num_nodes, hidden_layers_nodes[i - 1], hidden_layers_nodes[i],
                                           bias=use_bias, beta=use_beta))
            self.nodes.add_module(f'relu{i:d}', nn.ReLU(True))
        self.nodes.add_module(f'dense{len(hidden_layers_nodes):d}',
                              Linear4D(num_graphs, num_nodes, hidden_layers_nodes[-1], 1, bias=use_bias, beta=use_beta))
        if gumbel_nodes:
            self.nodes.add_module('nodeNoise', NodeNoise())
        self.nodes.add_module('sigmoid', nn.Sigmoid())

        if learning_roots:
            self.roots = Parameter(Tensor(num_graphs, num_nodes))
        else:
            self.register_parameter('roots', None)

        self.reset_parameters()

    def set_tau(self, t: float) -> None:
        self.tau = t

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.M_left)
        nn.init.uniform_(self.M_right)
        if self.learning_roots:
            nn.init.constant_(self.roots, 1)

    def forward(self, x: Tensor, temperature: float = None) -> Tuple[Tensor, Any]:
        if self.num_jumps is None:
            raise NotImplemented
        else:
            return self.forward_baseline(x, temperature)

    def extra_repr(self) -> str:
        return f'num_graphs: {self.num_graphs}, num_nodes={self.num_nodes}, num_leaves={self.num_leaves}, ' \
               f'num_jumps={self.num_leaves}, hidden_layers_nodes={"x".join(map(str, self.hidden_layers_nodes))}'

    def transition_matrix(self, x: Tensor, eps: float = None) -> Tuple[Tensor, Tensor]:
        b = self.nodes(x)
        if self.tau and self.training:
            M0 = F.gumbel_softmax(self.M_left, tau=self.tau, hard=False, dim=1)
            M1 = F.gumbel_softmax(self.M_right, tau=self.tau, hard=False, dim=1)
        else:
            M0 = torch.softmax(self.M_left, dim=1)
            M1 = torch.softmax(self.M_right, dim=1)
        if self.binarization_threshold:
            M0 = (M0 > self.binarization_threshold).float()
            M1 = (M1 > self.binarization_threshold).float()
        p = b * (M1 - M0).unsqueeze(1) + M0.unsqueeze(1)  # gb1n * q1mn

        q = torch.zeros(self.num_nodes + self.num_leaves, self.num_leaves, dtype=x.dtype, device=x.device)
        if eps is None:
            q[range(self.num_nodes, self.num_nodes + self.num_leaves), range(self.num_leaves)] = 1
        else:
            q[0, :] = eps
            q[range(self.num_nodes, self.num_nodes + self.num_leaves), range(self.num_leaves)] = 1 + eps
        return torch.cat([p, q.repeat(self.num_graphs, x.shape[0], 1, 1)], dim=-1), b

    def forward_baseline(self, x: Tensor, temperature: float = None) -> Tuple[Tensor, Tensor]:
        q, nodes_logits = self.transition_matrix(x)
        if temperature is not None:
            q = (1 - temperature) * q + temperature / (self.num_nodes + self.num_leaves)

        prob = torch.zeros(self.num_graphs, x.shape[0], self.num_nodes + self.num_leaves, dtype=x.dtype,
                           device=x.device)
        if self.roots is None:
            prob[..., 0] = 1
        else:
            prob[..., :self.num_nodes] = torch.softmax(self.roots, dim=1).unsqueeze(1)

        loss_nodes = 0
        z = []  # gbn
        for _ in range(self.num_jumps):
            if self.tau_regularization is not None:
                z.append(prob[..., :self.num_nodes])
            prob = torch.einsum('gbnm,gbm->gbn', q, prob)
        if self.tau_regularization is not None:
            z = torch.stack(z, dim=-2)  # gbsn
            z = self.interval(z)
            lam = torch.pow(2., -torch.arange(self.num_jumps, dtype=z.dtype, device=z.device))  # s
            z_sum = z.sum(dim=1)  # gsn
            alpha = (z * torch.pow(nodes_logits, self.tau_regularization)).sum(dim=1).div(
                    torch.where(z_sum == 0., torch.tensor(1, dtype=z.dtype, device=z.device), z_sum))  # gbsn * gb1n -> gsn
            alpha = self.interval(alpha)
            if self.tau_regularization == 1:
                c = (0.5 * (torch.log(alpha) + torch.log(1 - alpha))).sum(dim=-1)  # gs
            else:
                beta = (z * torch.pow(1 - nodes_logits, self.tau_regularization)).sum(dim=1).div(
                    torch.where(z_sum == 0., torch.tensor(1, dtype=z.dtype, device=z.device), z_sum))
                beta = self.interval(beta)
                c = (0.5 * (torch.log(alpha) + torch.log(beta))).sum(dim=-1)  # gs
            loss_nodes = (-lam.view(1, -1) * c).sum(dim=-1)  # g
        ret = prob[..., self.num_nodes:].permute(1, 2, 0)
        return self.interval(ret), loss_nodes  # gbn -> bng, g

    @staticmethod
    def interval(x: Tensor, eps: float = 1e-5) -> Tensor:
        values = torch.where(x > 0., torch.tensor(0, dtype=x.dtype, device=x.device), x - eps)
        x -= values
        values = torch.where(x < 1., torch.tensor(0, dtype=x.dtype, device=x.device), x - (1 - eps))
        x -= values
        return x


class PrototypeGraph(nn.Module):
    def __init__(self, hidden_layers_nodes: List[int], num_nodes: int, num_leaves: int, num_jumps: Optional[int],
                 num_graphs: int, use_representation: str = '', learning_roots: bool = False, tau: float = None,
                 use_bias: bool = True, use_beta: bool = False, gumbel_nodes: bool = False,
                 tau_regularization: float = None, binarization_threshold: float = 0) -> None:
        super().__init__()
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_leaves = num_leaves
        self.num_jumps = num_jumps

        if use_representation == '':
            self.representation = nn.Flatten(start_dim=1, end_dim=-1)
        else:
            self.representation = Net(50, False)
            if Path(use_representation).is_file():
                print("==> Loading pretrained model..")
                checkpoint = torch.load(use_representation, map_location='cpu')
                self.representation.load_state_dict(checkpoint)

        self.graph = Graphs(hidden_layers_nodes, num_nodes, num_leaves, num_jumps, learning_roots, num_graphs, tau,
                            use_bias, use_beta, gumbel_nodes, tau_regularization, binarization_threshold)

    def set_tau(self, t):
        self.graph.set_tau(t)

    def forward(self, x: Tensor, temperature: float = None) -> Tensor:
        x = self.representation(x)
        return self.graph(x, temperature)

    def trainable_parameters_representation(self, mode_representation: bool = True, mode_m: bool = True,
                                            mode_nodes: bool = True, mode_root: bool = True) -> Dict[str, List[Tensor]]:
        trainable_params = {'representation': [], 'm': [], 'nodes': [], 'root': []}
        for name, param in self.named_parameters():
            if name.startswith('representation.'):
                param.requires_grad = mode_representation
                if mode_representation:
                    trainable_params['representation'].append(param)
            elif name.startswith('graph.M_left') or name.startswith('graph.M_right'):
                param.requires_grad = mode_m
                if mode_m:
                    trainable_params['m'].append(param)
            elif name.startswith('graph.nodes'):
                param.requires_grad = mode_nodes
                if mode_nodes:
                    trainable_params['nodes'].append(param)
            elif name.startswith('graph.roots'):
                param.requires_grad = mode_root
                if mode_root:
                    trainable_params['root'].append(param)
        results = {}
        for key, val in trainable_params.items():
            if len(val):
                results[key] = val
        return results
