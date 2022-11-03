import math
from typing import Tuple, List

import numpy as np
from torch import Tensor, randperm, long
from torch.utils.data import Sampler


def num_iterations_update_scale(start_val: float, end_val: float, by_epoch: int, iters_per_epoch: int,
                                case_update: str) -> Tuple[int, float]:
    output = 0
    anneal_rate = 1.
    if case_update == 'linear':
        anneal_rate = 0.8
        output = int(
            math.floor(by_epoch * iters_per_epoch / (math.log10(end_val / start_val) / math.log10(anneal_rate))))
    elif case_update == 'exponential':
        anneal_rate = 1e-2
        output = int(math.floor(by_epoch * iters_per_epoch / (math.log(end_val / start_val) / -anneal_rate)))
    return output, anneal_rate


def mixup_data(x: Tensor, y: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.
    index = randperm(x.shape[0], dtype=x.dtype, device=x.device).to(long)
    mixed_x = lam * x + (1 - lam) * x[index, ...]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class BalancedBatchSampler(Sampler[int]):
    target: List[int]
    batch_size: int
    drop_last: bool
    shuffle: bool
    current_class: int

    def __init__(self, target: List[int], batch_size_per_class: int, drop_last: bool, shuffle: bool) -> None:
        self.target = target
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_size_per_class = batch_size_per_class

        sort_indexes = np.argsort(target)
        arr = np.array(target)[sort_indexes]
        self.classes, first_indexes, counts = np.unique(arr, return_index=True, return_counts=True)
        self.indexes = np.split(sort_indexes, first_indexes[1:])
        self.num_classes = self.classes.size
        self.batch_size = batch_size_per_class * self.num_classes

        assert self.batch_size < len(target), 'The size of a batch is greater than half of the whole data'

        self.current_class = 0
        self.current_index = np.empty(self.num_classes, dtype=int)
        self.current_index.fill(-1)
        self.max_elements_per_class = np.max(counts)

    def _index_shuffle(self, idx: int) -> None:
        np.random.shuffle(self.indexes[idx])

    def sampler(self):
        if self.shuffle:
            for i in range(self.num_classes):
                self._index_shuffle(i)

        while self.current_index[self.current_class] < self.max_elements_per_class - 1:
            self.current_index[self.current_class] += 1
            if self.current_index[self.current_class] == self.indexes[self.current_class].size and self.indexes[
                self.current_class].size < self.max_elements_per_class:
                if self.shuffle:
                    self._index_shuffle(self.current_class)
                self.current_index[self.current_class] = 0
            yield self.indexes[self.current_class][self.current_index[self.current_class]]
            self.current_class = (self.current_class + 1) % self.num_classes
        self.current_index[:] = -1

    def __iter__(self):
        batch = []
        for idx in self.sampler():
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield np.random.permutation(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield np.random.permutation(batch)

    def __len__(self):
        if self.drop_last:
            return self.max_elements_per_class // self.batch_size_per_class
        else:
            return (self.max_elements_per_class + self.batch_size_per_class - 1) // self.batch_size_per_class
