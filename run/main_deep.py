import argparse
from datetime import datetime
from functools import reduce
from pathlib import Path, PurePosixPath
from re import compile, search
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from song.deep.models.model import PrototypeGraph
from song.deep.utils import datasets
from song.deep.utils.resnet import names as models_names
from song.utils.loss import Loss
from song.utils.plot import plot2tensorboard
from song.utils.utils import num_iterations_update_scale, mixup_data, BalancedBatchSampler
from song.utils.utils_model import load_model, save_model, load_model_selected_graph, load_selected_weights


def parse_args(opt: Optional[List[str]]) -> Tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser(description="Training PrototypeGraph on CIFAR")
    parser.add_argument('--root_data', default='./data', help='Path to data')
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size used for training")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--dataset", default="CIFAR10", choices=datasets.names)
    parser.add_argument("--arch", default="ResNet18", choices=models_names)
    parser.add_argument("--lr", default=[0.1], type=float, nargs='+', help="learning rate")
    parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")

    # extra general options for main script
    parser.add_argument("--path_resume", default="", help="Overrides checkpoint path generation")
    parser.add_argument("--pretrained", action="store_true", help="Download pretrained model")

    parser.add_argument('--eps', type=float, default=None,
                        help='Value of epsilon for the case with infinite jumps step (default: None)')
    parser.add_argument('--teleportation', '-t', type=float, default=None, help='teleportation (default: None)')

    parser.add_argument('--update_scale', choices=['linear', 'exponential', 'constant'], default=None)
    parser.add_argument('--tau', type=float, default=None, help='non-negative scalar temperature for Gumbel-Softmax '
                                                                'if None use typical "softmax" (default: None)')
    parser.add_argument('--dirty_tau', action='store_true')
    parser.add_argument('--step_scheduler', type=int, default=10)

    parser.add_argument('--num_graphs', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--num_leaves', type=int, default=10)
    parser.add_argument('--num_jumps', type=int, default=None)
    parser.add_argument('--cross_entropy', action='store_true', help='Use cross-entropy loss (default: BCELoss)')
    parser.add_argument('--results', default='./results', help='Path to dictionary where will be save results.')
    parser.add_argument('--num_best_graph', type=int, default=1, help="Number best graphs for which model will be save")

    parser.add_argument("--no_trainable", default=[], choices=['resnet', 'm', 'nodes', 'root'], nargs='+')
    parser.add_argument("--no_resume_parts", default=[], choices=['representation', 'm', 'nodes', 'root'], nargs='+')
    parser.add_argument('--trainable_root', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    parser.add_argument('--use_mixup_data', action='store_true')
    parser.add_argument('--use_ensemble', action='store_true')
    parser.add_argument('--use_balanced_train_data', action='store_true')
    parser.add_argument('--prob_leaves_rate', type=float, default=0)
    parser.add_argument('--tau_regularization', type=float, default=None)
    parser.add_argument('--binarization_threshold', type=float, default=0)

    parser.add_argument('--scale_nodes_loss', type=float, default=None)
    parser.add_argument('--apply_nodes_loss', type=float, default=1)
    parser.add_argument('--use_beta', action='store_true')
    parser.add_argument('--gumbel_nodes', action='store_true')
    parser.add_argument('--use_distillation', action='store_true')
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--use_tqdm', action='store_true')
    if opt is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(opt)

    assert args.tau_regularization is None or 1 <= args.tau_regularization < 2
    assert 0 < args.apply_nodes_loss <= 1, 'Option "apply_nodes_loss" has value from interval [0, 1]'
    assert len(args.lr) in (
        1, 4), "Length of list learning rate must be 1 or 5 (for 'resnet', 'm', 'nodes', 'root')"
    assert args.teleportation is None or 0 < args.teleportation <= 0.5, \
        'Parameter "--teleportation | -t" should be between 0 and 0.5'

    args.use_distillation = args.use_distillation and args.pretrained
    info = f'{args.dataset}_nodes-{args.num_nodes}_leaves-{args.num_leaves}' \
           f'_N-{"inf" if args.num_jumps is None else args.num_jumps}_graphs-{args.num_graphs}' \
           f'_lr-{"-".join(map(str, args.lr))}_bs-{args.batch_size}{"_pretrained" if args.pretrained else ""}' \
           f'{"_frozen-" + "-".join(map(str, args.no_trainable)) if len(args.no_trainable) else ""}' \
           f'_loss-{"Cross-Entropy" if args.cross_entropy else "BCELoss"}' \
           f'{f"_probLeavesRate-{args.prob_leaves_rate:g}" if args.prob_leaves_rate else ""}' \
           f'{"_mixupData" if args.use_mixup_data else ""}' \
           f'{"_balancedTrainData" if args.use_balanced_train_data else ""}' \
           f'{"" if args.tau is None else f"_tau-{args.tau}"}{"-dirty" if args.dirty_tau else ""}' \
           f'{"_useBeta" if args.use_beta else ""}{"_gumbelNodes" if args.gumbel_nodes else ""}' \
           f'{f"_tau_regularization-{args.tau_regularization}" if args.tau_regularization is not None else ""}' \
           f'{f"_binaryM-{args.binarization_threshold}" if args.binarization_threshold else ""}' \
           f'{f"_addLossNodes-{args.scale_nodes_loss}-{args.apply_nodes_loss}" if args.scale_nodes_loss is not None else ""}' \
           f'{"_trainable-roots" if args.trainable_root else ""}{"_distillation" if args.use_distillation else ""}' \
           f'{"" if args.teleportation is None else f"_teleportation-{args.teleportation}"}' \
           f'{"" if args.update_scale is None else f"_updateScale-{args.update_scale}"}' \
           f'{"" if args.eps is None else f"_eps-{args.eps}"}{"_scheduler" if args.use_scheduler else ""}'
    return args, info


def learn_model(args: argparse.Namespace, info: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')

    if args.seed is None:  # 1234
        args.seed = np.random.randint(10, 10000, size=1)[0]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    print("==> Preparing data..")
    dataset_train = getattr(datasets, args.dataset)
    dataset_test = getattr(datasets, args.dataset)

    transform_train = dataset_train.transform_train()
    transform_test = dataset_test.transform_val()
    trainset = dataset_train(root=args.root_data, train=True, download=True, transform=transform_train)
    testset = dataset_test(root=args.root_data, train=False, download=True, transform=transform_test)
    kwargs = {}
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        kwargs.update({'num_workers': 4, 'pin_memory': True})

    if args.use_balanced_train_data:
        trainloader = DataLoader(
            trainset, batch_sampler=BalancedBatchSampler(trainset.targets, batch_size_per_class=args.batch_size,
                                                         drop_last=False, shuffle=True), **kwargs)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, **kwargs)

    assert args.num_leaves == len(trainset.classes)
    print(f"\033[0;34mTraining with dataset {args.dataset} and {len(trainset.classes)} classes\033[0m")
    print(f"\033[0;34mTesting with dataset {args.dataset} and {len(testset.classes)} classes\033[0m")

    start_epoch = 0
    if args.resume and str(PurePosixPath(args.path_resume).suffix) == '.csv':
        assert Path(args.path_resume).is_file()
        p = compile('(\w+)_nodes-(\d+)\w+leaves-(\d+)\w+N-(\d+)_graphs-(\d+)')
        m = search(p, args.path_resume)

        args.dataset = m.group(1)
        num_nodes = int(m.group(2))
        num_leaves = int(m.group(3))
        num_jumps = int(m.group(4))
        # num_graphs = int(m.group(5))
        num_graphs = 1

        info = info.replace(f'_nodes-{args.num_nodes}', f'_nodes-{num_nodes}')
        args.num_nodes = num_nodes
        info = info.replace(f'_leaves-{args.num_leaves}', f'_leaves-{num_leaves}')
        args.num_leaves = num_leaves
        info = info.replace(f'_N-{args.num_jumps}', f'_N-{num_jumps}')
        info = info.replace(f'_N-inf', f'_N-{num_jumps}')
        args.num_jumps = num_jumps
        info = info.replace(f'_graphs-{args.num_graphs}', f'_graphs-{num_graphs}')
        args.num_graphs = num_graphs

        df = pd.read_csv(args.path_resume, sep=',').sort_values(by='train_acc', ascending=False)
        args.id_graph = df.index.tolist()[0]
        args.path_resume = f'{PurePosixPath(args.path_resume).parent}/checkpoint/{PurePosixPath(PurePosixPath(args.path_resume).name).stem}/models-epoch{df.loc[args.id_graph, "train_epoch"]:03}.pth'

    net = PrototypeGraph(args.arch, args.pretrained, args.path_resume, device, args.num_nodes, args.num_leaves,
                         args.num_jumps, args.num_graphs, args.trainable_root, args.dataset, args.tau,
                         args.use_distillation, args.use_beta, args.gumbel_nodes, args.tau_regularization,
                         args.binarization_threshold)

    if args.resume:
        assert Path(args.path_resume).is_file()
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        if hasattr(args, 'id_graph'):
            # net, checkpoint = load_model_selected_graph(net, args.path_resume, args.id_graph, device)
            net, checkpoint = load_selected_weights(net, args.path_resume, args.id_graph, args.no_resume_parts, device)
        else:
            net, _, checkpoint = load_model(net, None, args.path_resume, device)

        print(f"\033[0;35mParams resume:\033[0m")
        for key, val in checkpoint.items():
            print(f"\033[0;35m{key}: {val}\033[0m")

    net = net.to(device)

    trainable_params = net.trainable_parameters_representation(
        'resnet' not in args.no_trainable, 'm' not in args.no_trainable, 'nodes' not in args.no_trainable,
        'root' not in args.no_trainable)

    print(f"\033[0;33m{'=' * 50}\033[0m")
    print(f"\033[0;33mTrainable parameters:\033[0m")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(f"\033[0;33m{name}\033[0m")
    print(f"\033[0;33m{'=' * 50}\033[0m")

    params = []
    if len(args.lr) == 1:
        for _, p in trainable_params.items():
            params += p
        params = [{'params': params, 'lr': args.lr[0]}]
    else:
        lr = {'resnet': args.lr[0], 'm': args.lr[1], 'nodes': args.lr[2], 'root': args.lr[3]}
        print('\033[0;35mLearning rate for trainable parameters:\033[0m')
        for key, param in trainable_params.items():
            print(f'\033[0;35m{key}: {lr[key]}\033[0m')
            params.append({'params': trainable_params[key], 'lr': lr[key]})
    optimizer = optim.Adam(params)

    criterion = Loss(args.cross_entropy, len(trainset.classes), args.num_graphs)
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_scheduler, gamma=0.9)

    scalar2update = 1
    teleportation = args.teleportation
    args.temp_interval = idx_scheduler = 0
    if args.tau is not None and args.update_scale is not None:
        start_val = args.tau
        finish_val = 0.3
        args.temp_interval, anneal_rate = num_iterations_update_scale(args.tau, finish_val, int(0.5 * args.epochs),
                                                                      len(trainloader), args.update_scale)
    if args.teleportation is not None and args.update_scale is not None:
        start_val = args.teleportation
        finish_val = 1e-4
        args.temp_interval, anneal_rate = num_iterations_update_scale(
            args.teleportation, finish_val, int(0.3 * args.epochs), len(trainloader), args.update_scale)

    print('=' * 50)
    print('Parameters:')
    for arg in vars(args):
        if getattr(args, arg) not in [None, False]:
            print(f'{arg}: \033[0;1;32m{getattr(args, arg)}\033[0m')
    print('=' * 50)

    info += f'_seed-{args.seed}_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}'
    path_tensorboard = f'{args.results}/tensorboard/{info}'
    Path(path_tensorboard).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(path_tensorboard)
    dir_checkpoint = f'{args.results}/checkpoint/{info}'
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    ####################################
    #          learning model          #
    ####################################
    scores_dict = {'test_loss': np.full(args.num_graphs, np.inf), 'test_acc': np.full(args.num_graphs, 0.),
                   'test_epoch': np.full(args.num_graphs, 0), 'train_loss': np.full(args.num_graphs, np.inf),
                   'train_acc': np.full(args.num_graphs, 0.), 'train_epoch': np.full(args.num_graphs, 0),
                   'max_acc': np.full(args.num_graphs, 0.), 'max_acc_epoch': np.full(args.num_graphs, 0),
                   'ensemble_acc': 0, 'ensemble_epoch': 0, 'binary_ensemble_acc': 0, 'binary_ensemble_epoch': 0
                   }
    saved_epochs = []

    ####################################
    #          validation step         #
    ####################################
    net.eval()

    tst_acc = total = tst_binary_ensemble_acc = tst_ensemble_acc = 0
    tst_tqdm = tqdm(enumerate(testloader, 0), total=len(testloader), desc='Test step',
                    leave=False) if args.use_tqdm else enumerate(testloader, 0)
    with torch.no_grad():
        for i, (data, label) in tst_tqdm:
            data = data.to(device)
            label = label.to(device)

            output, _ = net(data, eps=args.eps)
            _, predicted = torch.max(output[0], 1, keepdim=True)
            tst_acc += (predicted.squeeze(1) == label.unsqueeze(1).repeat(1, args.num_graphs)).sum(0).detach()

            if args.use_ensemble:
                binary_output = torch.zeros_like(output[0]).scatter_(1, predicted.to(torch.long), 1)
                sum_prob = binary_output.sum(-1)
                _, predicted = torch.max(sum_prob, 1)
                tst_binary_ensemble_acc += (predicted == label).sum(0).detach()

                sum_prob = output[0].sum(-1)
                _, predicted = torch.max(sum_prob, 1)
                tst_ensemble_acc += (predicted == label).sum(0).detach()

            total += label.size(0)

    tst_acc = tst_acc.cpu().numpy() / total
    if args.use_ensemble:
        tst_binary_ensemble_acc = tst_binary_ensemble_acc.item() / total
        tst_ensemble_acc = tst_ensemble_acc.item() / total
        print(f'\033[0;1;33mINIT ACC: {tst_acc.mean():.4f}, binary ensemble: {tst_binary_ensemble_acc:.4f}, '
              f'ensemble: {tst_ensemble_acc:.4f}\033[0m')
    else:
        print(f'\033[0;1;33mINIT ACC: {tst_acc.mean():.4f}\033[0m')
    writer.add_scalars('test_acc', dict(zip([f'graph-{i}' for i in range(args.num_graphs)], tst_acc)), -1)

    turn_on = 40
    assert turn_on < args.epochs
    p = None
    stop_nodes_loss = int(args.epochs * args.apply_nodes_loss)
    if args.use_tqdm:
        print(f'==> Save: \033[0;35m{info}\033[0m')
        epoch_tqdm = tqdm(range(start_epoch, args.epochs), desc="Training")
    else:
        epoch_tqdm = range(start_epoch, args.epochs)
    for epoch in epoch_tqdm:
        ####################################
        #            train step            #
        ####################################
        net.train()

        trn_loss = 0
        trn_tqdm = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc='Train step',
                        leave=False) if args.use_tqdm else enumerate(trainloader, 0)
        for i, (data, label) in trn_tqdm:
            # ===============================================
            if args.dirty_tau:
                p = (epoch * len(trainloader) + i) / (args.epochs * len(trainloader))
                p = np.random.choice(2, 1, p=[p, 1 - p])[0]
                if p == 0:
                    net.set_tau(None)
                else:
                    net.set_tau(args.tau)
            # ===============================================
            data = data.to(device)
            label = label.to(device)
            if args.use_mixup_data:
                data, targets_a, targets_b, lam = mixup_data(data, label, 0.5)

            # ===================forward=====================
            output, distillation_target = net(data, teleportation, args.eps)
            trn_prob_leaves = torch.log(output[0].sum(1)).mean(0)
            if args.use_distillation:
                # label = torch.argmax(distillation_target, 1)
                label = torch.softmax(distillation_target, dim=1)
            if args.use_mixup_data:
                loss = criterion.mixup_forward(output[0], targets_a, targets_b, lam)
            else:
                loss = criterion(output[0], label)
            scale = args.prob_leaves_rate
            if scale > 1:
                scale *= max(0, (epoch - turn_on) * len(trainloader) + i) / ((args.epochs - turn_on) * len(trainloader))
            loss -= trn_prob_leaves * scale
            if args.tau_regularization is not None:
                loss_nodes = output[1]
                if args.scale_nodes_loss is not None:
                    loss += args.scale_nodes_loss * loss_nodes

            mean_loss = loss.mean()
            # ===================backward====================
            optimizer.zero_grad(set_to_none=True)

            assert torch.isfinite(mean_loss), '\033[1;31mLoss has Nan or Inf\033[0m'
            mean_loss.backward()
            optimizer.step()

            if args.update_scale is not None and args.update_scale != 'constant' and (
                    epoch * len(trainloader) + i + 1) % args.temp_interval == 0:
                if args.update_scale == 'linear':
                    scalar2update *= anneal_rate
                elif args.update_scale == 'exponential':
                    scalar2update = start_val * np.exp(-anneal_rate * idx_scheduler)
                if scalar2update < finish_val:
                    args.update_scale = None
                    if args.tau is not None:
                        scalar2update = finish_val
                    if args.teleportation is not None:
                        scalar2update = None
                idx_scheduler += 1
                if args.tau is not None:
                    net.set_tau(scalar2update)
                    writer.add_scalar('gumbel_tau', net.graph.tau, epoch * len(trainloader) + i)
                if args.teleportation is not None:
                    teleportation = scalar2update
                    writer.add_scalar('teleportation', teleportation, epoch * len(trainloader) + i)

            # ===================logger========================
            if args.tau_regularization is not None:
                writer.add_scalars('train_loss_nodes',
                                   dict(zip([f'graph-{i}' for i in range(args.num_graphs)], loss_nodes)),
                                   epoch * len(trainloader) + i)
            writer.add_scalars('train_loss',
                               dict(zip([f'graph-{i}' for i in range(args.num_graphs)], loss)),
                               epoch * len(trainloader) + i)
            writer.add_scalars('train_prob_leaves',
                               dict(zip([f'graph-{i}' for i in range(args.num_graphs)], trn_prob_leaves)),
                               epoch * len(trainloader) + i)
            if args.use_tqdm:
                trn_tqdm.set_description(
                    f"Train loss: {mean_loss:.5f}{'' if p is None else f'(tau: {p * args.tau})'}"
                    f"{'' if teleportation is None else f', teleportation: {teleportation:.5f}'}")
            trn_loss += loss.detach()
        trn_loss = trn_loss.cpu().numpy() / len(trainloader)

        if args.use_scheduler:
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        ####################################
        #          validation step         #
        ####################################
        net.eval()

        tst_loss_graph = tst_loss_nodes = tst_acc = prob_leaves = total = tst_binary_ensemble_acc = tst_ensemble_acc = 0
        tst_tqdm = tqdm(enumerate(testloader, 0), total=len(testloader), desc='Test step',
                        leave=False) if args.use_tqdm else enumerate(testloader, 0)
        with torch.no_grad():
            for i, (data, label) in tst_tqdm:
                data = data.to(device)
                label = label.to(device)

                output, _ = net(data, eps=args.eps)
                loss_graph = criterion(output[0], label)
                tst_loss_graph += loss_graph.detach()
                if args.tau_regularization is not None:
                    tst_loss_nodes += output[1].detach()
                prob_leaves += output[0].sum(dim=(0, 1)).detach()

                _, predicted = torch.max(output[0], 1, keepdim=True)
                tst_acc += (predicted.squeeze(1) == label.unsqueeze(1).repeat(1, args.num_graphs)).sum(0).detach()

                if args.use_ensemble:
                    binary_output = torch.zeros_like(output[0]).scatter_(1, predicted.to(torch.long), 1)
                    sum_prob = binary_output.sum(-1)
                    _, predicted = torch.max(sum_prob, 1)
                    tst_binary_ensemble_acc += (predicted == label).sum(0).detach()

                    sum_prob = output[0].sum(-1)
                    _, predicted = torch.max(sum_prob, 1)
                    tst_ensemble_acc += (predicted == label).sum(0).detach()

                total += label.size(0)

        tst_loss_graph = tst_loss_graph.cpu().numpy() / total
        tst_acc = tst_acc.cpu().numpy() / total
        if args.use_ensemble:
            tst_binary_ensemble_acc = tst_binary_ensemble_acc.item() / total
            tst_ensemble_acc = tst_ensemble_acc.item() / total
            writer.add_scalar('test_binary_ensemble_acc', tst_binary_ensemble_acc, epoch)
            writer.add_scalar('test_ensemble_acc', tst_ensemble_acc, epoch)
        prob_leaves = prob_leaves.cpu().numpy() / total
        if args.tau_regularization is not None:
            tst_loss_nodes = tst_loss_nodes.cpu().numpy() / total
            writer.add_scalars('test_loss_nodes', dict(zip([f'graph-{i}' for i in range(args.num_graphs)],
                                                           tst_loss_nodes)), epoch)
        writer.add_scalars('test_loss_graph', dict(zip([f'graph-{i}' for i in range(args.num_graphs)], tst_loss_graph)),
                           epoch)
        writer.add_scalars('test_acc', dict(zip([f'graph-{i}' for i in range(args.num_graphs)], tst_acc)), epoch)
        writer.add_scalars('test_prob_leaves', dict(zip([f'graph-{i}' for i in range(args.num_graphs)], prob_leaves)),
                           epoch)
        if args.tau_regularization is not None:
            tst_loss = tst_loss_graph
            if args.scale_nodes_loss is not None:
                tst_loss += args.scale_nodes_loss * tst_loss_nodes
            writer.add_scalars('test_loss', dict(zip([f'graph-{i}' for i in range(args.num_graphs)], tst_loss)), epoch)

        ####################################
        #       logger, save model         #
        ####################################
        if args.use_tqdm:
            epoch_tqdm.set_description(
                f"Train graph loss: {trn_loss.mean():.5f}, test: {tst_loss_graph.mean():.5f} "
                f"| acc: {tst_acc.mean():.3f}, prob_leaves: {prob_leaves.mean():.3f}"
            )
        else:
            print(f'Epoch {epoch}|{args.epochs}, train loss: {trn_loss.mean():.5f}, '
                  f'test loss: {tst_loss_graph.mean():.5f} '
                  f'| acc: {tst_acc.mean():.3f}, prob_leaves: {prob_leaves.mean():.3f} '
                  f'(minimal test-loss: {np.mean(scores_dict["test_loss"]):.5f}) - '
                  f'teleportation: {teleportation if teleportation is None else f"{teleportation:.5f}"}')

        idx = trn_loss < scores_dict['train_loss']
        scores_dict['train_loss'][idx] = trn_loss[idx]
        scores_dict['train_acc'][idx] = tst_acc[idx]
        scores_dict['train_epoch'][idx] = epoch

        idx = tst_loss_graph < scores_dict['test_loss']
        scores_dict['test_loss'][idx] = tst_loss_graph[idx]
        scores_dict['test_acc'][idx] = tst_acc[idx]
        scores_dict['test_epoch'][idx] = epoch

        idx = tst_acc > scores_dict['max_acc']
        scores_dict['max_acc'][idx] = tst_acc[idx]
        scores_dict['max_acc_epoch'][idx] = epoch

        if args.use_ensemble:
            if tst_ensemble_acc > scores_dict['ensemble_acc']:
                scores_dict['ensemble_acc'] = tst_ensemble_acc
                scores_dict['ensemble_epoch'] = epoch

            if tst_binary_ensemble_acc > scores_dict['binary_ensemble_acc']:
                scores_dict['binary_ensemble_acc'] = tst_binary_ensemble_acc
                scores_dict['binary_ensemble_epoch'] = epoch

            best_epoch = reduce(np.union1d, (
                scores_dict['train_epoch'][np.argsort(scores_dict['train_acc'])[-args.num_best_graph:]],
                scores_dict['test_epoch'][np.argsort(scores_dict['test_acc'])[-args.num_best_graph:]],
                scores_dict['max_acc_epoch'][np.argsort(scores_dict['max_acc'])[-args.num_best_graph:]],
                [scores_dict['ensemble_epoch'], scores_dict['binary_ensemble_epoch']]
            ))
        else:
            best_epoch = reduce(np.union1d, (
                scores_dict['train_epoch'][np.argsort(scores_dict['train_acc'])[-args.num_best_graph:]],
                scores_dict['test_epoch'][np.argsort(scores_dict['test_acc'])[-args.num_best_graph:]],
                scores_dict['max_acc_epoch'][np.argsort(scores_dict['max_acc'])[-args.num_best_graph:]]
            ))
        if epoch in best_epoch:
            save_model(net, None, f'{dir_checkpoint}/models-epoch{epoch:03}.pth', epoch=epoch, acc=tst_acc)
            saved_epochs.append(epoch)
            for id_ep in np.setdiff1d(saved_epochs, best_epoch):
                Path(f'{dir_checkpoint}/models-epoch{id_ep:03}.pth').unlink(missing_ok=True)
                saved_epochs.remove(id_ep)

        ####################################
        #           scheduler              #
        ####################################
        if args.use_scheduler:
            scheduler.step()

        if args.scale_nodes_loss is not None and epoch == stop_nodes_loss:
            print(f'\n\033[0;1;31mTurning off nodes loss!\033[0m')
            args.scale_nodes_loss = None

        if (epoch + 1) % 10 == 0:
            plot2tensorboard(net, writer, epoch)
            df = pd.DataFrame.from_dict(scores_dict)
            df.to_csv(f'{args.results}/{info}.csv', index=False)

    writer.close()
    df = pd.DataFrame.from_dict(scores_dict)
    df.to_csv(f'{args.results}/{info}.csv', index=False)
    print(tabulate(tabular_data=df.sort_values(by='train_acc', ascending=False), headers="keys", tablefmt="pretty",
                   floatfmt=".4f"))


def evaluation_model(args: argparse.Namespace) -> None:
    print('\033[0;1;33mEvaluating PrototypeGraph on CIFAR\033[0m')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')

    # Data
    print("==> Preparing data..")
    dataset_test = getattr(datasets, args.dataset)

    transform_test = dataset_test.transform_val()
    testset = dataset_test(root=args.root_data, train=False, download=True, transform=transform_test)
    kwargs = {}
    if device.type == 'cuda':
        kwargs.update({'num_workers': 4, 'pin_memory': True})
    testloader = DataLoader(testset, batch_size=100, shuffle=False, **kwargs)
    print(f"\033[0;34mTesting with dataset {args.dataset} and {len(testset.classes)} classes\033[0m")

    assert Path(args.path_resume).is_file()
    p = compile('(\w+)_nodes-(\d+)\w+leaves-(\d+)\w+N-(\d+)_graphs-(\d+)')
    m = search(p, args.path_resume)

    args.dataset = m.group(1)
    args.num_nodes = int(m.group(2))
    args.num_leaves = int(m.group(3))
    args.num_jumps = int(m.group(4))
    args.num_graphs = int(m.group(5))

    p = compile('CIFAR(\d+)')
    m = search(p, args.dataset)
    args.root_data = f'{args.root_data}/cifar-{int(m.group(1))}'

    for arg in vars(args):
        print(f'{arg}: \033[0;1;32m{getattr(args, arg)}\033[0m')

    net = PrototypeGraph(args.arch, False, args.path_resume, device, args.num_nodes, args.num_leaves,
                         args.num_jumps, args.num_graphs, args.trainable_root, args.dataset, None,
                         args.use_distillation)
    net = net.to(device)

    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    net, checkpoint = load_model_selected_graph(net, args.path_resume, args.id_graph, device)
    net.eval()

    print(f"\033[0;35mParams resume:\033[0m")
    for key, val in checkpoint.items():
        print(f"\033[0;35m{key}: {val}\033[0m")

    tst_acc = total = 0
    with torch.no_grad():
        for i, (data, label) in tqdm(enumerate(testloader, 0), total=len(testloader), desc='Evaluation', leave=False):
            data = data.to(device)
            label = label.to(device)

            prob, _ = net(data)
            _, predicted = torch.max(prob, 1)

            tst_acc += (predicted == label.unsqueeze(1).repeat(1, args.num_graphs)).sum(0).detach()
            total += label.size(0)

    tst_acc = tst_acc.cpu().numpy() / total
    print(f'Accuracy: {tst_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrototypeGraph')
    parser.add_argument('--evaluate', '-e', action='store_true', help='The run evaluation training model')
    args, unknown = parser.parse_known_args()
    ops, info = parse_args(unknown)
    if args.evaluate:
        evaluation_model(ops)
    else:
        learn_model(ops, info)
