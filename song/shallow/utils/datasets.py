from abc import ABC
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Connect4(Dataset, ABC):
    convert_data = dict(zip(*pd.factorize(['b', 'x', 'o'], sort=True)))
    classes = dict(zip(*pd.factorize(['win', 'loss', 'draw'], sort=True)))

    def __init__(self, root: str, train: bool = True) -> None:
        super(Connect4, self).__init__()

        self.num_classes = len(self.classes)
        self.train = train

        data = pd.read_csv(f'{root}/Connect-4/connect-4.data', sep=',', header=None)

        targets = pd.factorize(data.iloc[:, -1], sort=True)[0]

        data = pd.get_dummies(data.iloc[:, :-1])
        data = data.to_numpy()

        size = data.shape[0]
        permutation = np.random.RandomState(1234).permutation(size)
        cut = int(0.9 * size)

        if self.train:
            self.data = data[permutation[:cut]]
            self.targets = targets[permutation[:cut]]
        else:
            self.data = data[permutation[cut:]]
            self.targets = targets[permutation[cut:]]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        vector = torch.tensor(self.data[index])
        target = torch.tensor(self.targets[index])
        return vector.float(), target

    def __len__(self):
        return self.targets.size


class Letter(Dataset, ABC):
    classes = dict(zip(*pd.factorize(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
        sort=True)))

    def __init__(self, root: str, train: bool = True) -> None:
        super(Letter, self).__init__()

        self.num_classes = len(self.classes)
        self.train = train

        data = pd.read_csv(f'{root}/Letter/letter.data', sep=',', header=None)
        data = data.apply(lambda col: pd.factorize(col, sort=True)[0]).to_numpy()
        size = data.shape[0]
        cut = int(0.9 * size)

        if self.train:
            self.data = data[:cut, :-1]
            self.targets = data[:cut, -1]
        else:
            self.data = data[cut:, :-1]
            self.targets = data[cut:, -1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        vector = torch.tensor(self.data[index])
        target = torch.tensor(self.targets[index])
        return vector.float(), target

    def __len__(self):
        return self.targets.size


class EEG(Dataset, ABC):
    classes = [0, 1]

    def __init__(self, root: str, train: bool = True, normalize: bool = True) -> None:
        super(EEG, self).__init__()

        self.num_classes = len(self.classes)
        self.train = train

        data, targets = [], []
        filename = f'{root}/EEG/EEG Eye State.arff'
        with open(filename, 'r') as file:
            lines = file.read().split('@DATA')[1].split('\n')
            for line in lines:
                if line:
                    vals = line.split(',')
                    data.append([float(v) for v in vals[:-1]])
                    targets.append(int(vals[-1]))
        size = len(data)
        data = np.array(data)
        targets = np.array(targets)

        permutation = np.random.RandomState(1234).permutation(size)
        cut = int(0.9 * size)

        if self.train:
            self.data = data[permutation[:cut]]
            self.targets = targets[permutation[:cut]]
        else:
            self.data = data[permutation[cut:]]
            self.targets = targets[permutation[cut:]]

        if normalize:
            mean = data[permutation[:cut]].mean(axis=0)
            std = data[permutation[:cut]].std(axis=0)
            self.data = (self.data - mean) / std

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        vector = torch.tensor(self.data[index])
        target = torch.tensor(self.targets[index])
        return vector.float(), target

    def __len__(self):
        return self.targets.size
