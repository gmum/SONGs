import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, use_cross_entropy: bool = False, num_classes: int = None, num_graphs: int = None) -> None:
        super(Loss, self).__init__()

        self.use_cross_entropy = use_cross_entropy
        self.num_classes = num_classes
        self.num_graphs = num_graphs
        if use_cross_entropy:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criterion = nn.BCELoss(reduction='none')

    def forward(self, input_pred: Tensor, target: Tensor) -> Tensor:
        assert target.dim() <= 2
        if self.use_cross_entropy:
            if target.dim() == 2:
                _, target = torch.max(target, 1)
            if self.num_graphs:
                target = target.unsqueeze(-1).repeat(1, self.num_graphs)
                ret = self.criterion(input_pred, target).mean(dim=0)
            else:
                ret = self.criterion(input_pred, target)
        else:
            if target.dim() == 1:
                assert self.num_classes
                target = F.one_hot(target, num_classes=self.num_classes).float()

            scale = torch.ones_like(target)
            _, index = torch.max(target, dim=1, keepdim=True)
            scale.scatter_(1, index, self.num_classes - 1)
            scale /= self.num_classes - 1

            if self.num_graphs:
                target = target.unsqueeze(-1).repeat(1, 1, self.num_graphs)
                ret = self.criterion(input_pred, target) * scale.unsqueeze(-1)
                ret = ret.mean(dim=(0, 1))
            else:
                ret = self.criterion(input_pred, target) * scale
        return ret

    def mixup_forward(self, pred: Tensor, target_a: Tensor, target_b: Tensor, lam: float) -> Tensor:
        return lam * self.forward(pred, target_a) + (1 - lam) * self.forward(pred, target_b)
