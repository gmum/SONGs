"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

__all__ = names = ("ResNet10", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152")

model_urls = {
    (
        "ResNet10",
        "CIFAR10",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-ResNet10.pth",
    (
        "ResNet10",
        "CIFAR100",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR100-ResNet10.pth",
    (
        "ResNet18",
        "CIFAR10",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-ResNet18.pth",
    (
        "ResNet18",
        "CIFAR100",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR100-ResNet18.pth",
    (
        "ResNet18",
        "TinyImagenet200",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-ResNet18.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_output_classes=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if use_output_classes:
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        if use_output_classes:
            print('\033[0;1;31mThe model returns the output of classification (the last layer of ResNet model)\033[0m')
        self.use_output_classes = use_output_classes
        self.size_of_feature = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[2:])  # global average pooling
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out_features = self.features(x)
        if self.use_output_classes:
            out_classes = self.linear(out_features)
        else:
            out_classes = None
        return out_features, out_classes


def coerce_state_dict(state_dict, reference_state_dict):
    print("\033[0;1;36mInformation form loading model: \033[0m", end="")
    for key, val in state_dict.items():
        if key in ['net', 'model_state_dict']:
            state_dict = val
        else:
            print(f"\033[0;1;36m{key}: {val}\033[0m", end=", ")
    print()
    has_reference_module = list(reference_state_dict)[0].startswith("module.")
    has_module = list(state_dict)[0].startswith("module.")
    if not has_reference_module and has_module:
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    elif has_reference_module and not has_module:
        state_dict = {"module." + key: value for key, value in state_dict.items()}
    return state_dict


def _ResNet(arch, *args, pretrained=False, progress=True, dataset="CIFAR10", **kwargs):
    model = ResNet(*args)
    if pretrained:
        assert 'root' in kwargs.keys() and 'device' in kwargs.keys()

        valid_keys = [key for key in [(arch, dataset)] if key in model_urls]
        if not valid_keys:
            raise UserWarning(f"None of the keys {(arch, dataset)} correspond to a pretrained model.")
        key = valid_keys[-1]
        print(f"\033[0;1;32mLoading pretrained model {key} from {model_urls[key]}\033[0m")

        Path(kwargs['root']).mkdir(parents=True, exist_ok=True)

        state_dict = load_state_dict_from_url(
            url=model_urls[key],
            model_dir=kwargs['root'],
            map_location=kwargs['device'],
            progress=progress,
            check_hash=False
        )

        state_dict = coerce_state_dict(state_dict, model.state_dict())
        if 'load_linear_weights' in kwargs and kwargs['load_linear_weights']:
            state_dict_selection = {}
            add_weigths = {}
            for key, val in state_dict.items():
                if key in model.state_dict():
                    state_dict_selection[key] = val
                    # if key.startswith('linear.'):
                    #     add_weigths[key] = val
                else:
                    add_weigths[key] = val
            model.load_state_dict(state_dict_selection)
            return model, {'weight': add_weigths['linear.weight'], 'bias': add_weigths['linear.bias']}
        else:
            state_dict = {key: state_dict[key] for key, val in model.state_dict().items()}
            model.load_state_dict(state_dict)
            return model
    else:
        return model


def ResNet10(num_classes, use_output_classes, pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet10",
        BasicBlock,
        [1, 1, 1, 1],
        num_classes,
        use_output_classes,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet18(num_classes, use_output_classes, pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet18",
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        use_output_classes,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet34(num_classes, use_output_classes, pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet34",
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        use_output_classes,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet50(num_classes, use_output_classes, pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet50",
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        use_output_classes,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet101(num_classes, use_output_classes, pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet101",
        Bottleneck,
        [3, 4, 23, 3],
        num_classes,
        use_output_classes,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet152(num_classes, use_output_classes, pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet152",
        Bottleneck,
        [3, 8, 36, 3],
        num_classes,
        use_output_classes,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )
