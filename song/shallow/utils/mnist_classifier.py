import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self, dim_representation=50, use_output_classes=True):
        super().__init__()
        self.dim_representation = dim_representation
        self.use_output_classes = use_output_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 4 * 4, dim_representation)
        )
        self.fc = nn.Linear(dim_representation, 10)

    def forward(self, x):
        out_features = self.features(x)
        if self.use_output_classes:
            return self.fc(out_features)
        else:
            return out_features


def train(model, device, train_loader, optimizer, criterion, writer_tensorboard, epoch):
    model.train()
    tqdm_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (data, target) in tqdm_loop:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(torch.sigmoid(output), F.one_hot(target, num_classes=10).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm_loop.set_description(f'Loss: {loss.item():.6f}')
        writer_tensorboard.add_scalar('train_loss', loss, epoch * len(train_loader) + batch_idx)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader), leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(torch.sigmoid(output), F.one_hot(target, num_classes=10).float())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--root_data', default='./data', help='Path to data')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_model', default=None, help='Path to director')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'\033[0;1;31mDevice={device.type}\033[0m')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': 100}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset1 = datasets.MNIST(args.root_data, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(args.root_data, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    tqdm_epochs = tqdm(range(args.epochs))

    info = f'lr_{args.lr}_bs-{args.batch_size}_gamma-{args.gamma}' \
           f'_seed-{args.seed}_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}'
    path_tensorboard = f'{args.save_model}/tensorboard/{info}'
    Path(path_tensorboard).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(path_tensorboard)
    dir_checkpoint = f'{args.save_model}/checkpoint/{info}'
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    max_acc = 0
    for epoch in tqdm_epochs:
        train(model, device, train_loader, optimizer, criterion, writer, epoch)
        tst_loss, acc = test(model, device, test_loader, criterion)
        scheduler.step()
        tqdm_epochs.set_description(f'Test loss: {tst_loss:.6f}, acc: {acc:.3f} (max acc: {max_acc:.3f})')
        writer.add_scalar('test_loss', tst_loss, epoch)
        writer.add_scalar('test_acc', acc, epoch)

        if max_acc < acc:
            max_acc = acc
            if args.save_model is not None:
                torch.save(model.state_dict(), f"{dir_checkpoint}/mnist_cnn.pt")
    writer.close()


if __name__ == '__main__':
    main()
