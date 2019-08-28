# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import argparse
import copy
import datetime
import os
import time


from IPython import embed
from architectures import ARCHITECTURES, get_architecture
from attacks import Attacker, PGD_L2, DDN
from datasets import get_dataset, DATASETS, get_num_classes,\
                     MultiDatasetsDataLoader, TiTop50KDataset
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_

LABELLED = 0
PSEUDO_LABELLED = 1

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--pretrained-model', type=str, default='',
                    help='Path to a pretrained model')
parser.add_argument('--use-unlabelled', action='store_true',
                    help='Use unlabelled data via self-training.')
parser.add_argument('--self-training-weight', type=float, default=1.0,
                    help='Weight of self-training.')


#####################
# Attack params
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--attack', default='DDN', type=str, choices=['DDN', 'PGD'])
parser.add_argument('--epsilon', default=64.0, type=float)
parser.add_argument('--num-steps', default=10, type=int)
parser.add_argument('--warmup', default=1, type=int, help="Number of epochs over which \
-                    the maximum allowed perturbation increases linearly from zero to args.epsilon.")
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
parser.add_argument('--train-multi-noise', action='store_true', 
                    help="if included, the weights of the network are optimized using all the noise samples. \
-                       Otherwise, only one of the samples is used.")
parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")

# PGD-specific
parser.add_argument('--random-start', default=True, type=bool)

# DDN-specific
parser.add_argument('--init-norm-DDN', default=256.0, type=float)
parser.add_argument('--gamma-DDN', default=0.05, type=float)


args = parser.parse_args()

args.epsilon /= 256.0
args.init_norm_DDN /= 256.0

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Copies files to the outdir to store complete script with each experiment
    copy_code(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    labelled_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                      num_workers=args.workers, pin_memory=pin_memory)
    if args.use_unlabelled:
        pseudo_labelled_loader = DataLoader(TiTop50KDataset(), shuffle=True, batch_size=args.batch,
                                  num_workers=args.workers, pin_memory=pin_memory)
        train_loader = MultiDatasetsDataLoader([labelled_loader, pseudo_labelled_loader])
    else:
        train_loader = MultiDatasetsDataLoader([labelled_loader])
    
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    if args.pretrained_model != '':
        assert args.arch == 'cifar_resnet110', 'Unsupported architecture for pretraining'
        checkpoint = torch.load(args.pretrained_model)
        model = get_architecture(checkpoint["arch"], args.dataset)
        model.load_state_dict(checkpoint['state_dict'])
        model[1].fc = nn.Linear(64, get_num_classes('cifar10')).cuda()
    else:
        model = get_architecture(args.arch, args.dataset)

    if args.attack == 'PGD':
        print('Attacker is PGD')
        attacker = PGD_L2(steps=args.num_steps, device='cuda', max_norm=args.epsilon)
    elif args.attack == 'DDN':
        print('Attacker is DDN')
        attacker = DDN(steps=args.num_steps, device='cuda', max_norm=args.epsilon, 
                    init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)
    else:
        raise Exception('Unknown attack')

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    starting_epoch = 0
    logfilename = os.path.join(args.outdir, 'log.txt')

    # Load latest checkpoint if exists (to handle philly failures) 
    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    if args.resume:
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)
            starting_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                         .format(model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
            if args.adv_training:
                init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttrainacc\ttestacc\ttestaccNor")
            else:
                init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttrainacc\ttestacc")
    else:
        if args.adv_training:
            init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttrainacc\ttestacc\ttestaccNor")
        else:
            init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttrainacc\ttestacc")

    for epoch in range(starting_epoch, args.epochs):
        scheduler.step(epoch)
        attacker.max_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon/args.warmup])
        attacker.init_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon/args.warmup])

        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd, attacker)
        test_loss, test_acc, test_acc_normal = test(test_loader, model, criterion, args.noise_sd, attacker)
        after = time.time()

        if args.adv_training:
            log(logfilename, "{}\t{:.2f}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before,
                scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc, test_acc_normal))
        else:
            log(logfilename, "{}\t{:.2f}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before,
                scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)


def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, 
        epoch: int, noise_sd: float, attacker: Attacker=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    requires_grad_(model, True)

    for i, (batch, dataLoaderIdx) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)     

        mini_batches = get_minibatches(batch, args.num_noise_vec)
        noisy_inputs_list = []
        for inputs, targets in mini_batches:
            inputs = inputs.cuda()
            targets = targets.cuda()

            inputs = inputs.repeat((1, args.num_noise_vec, 1, 1)).view(batch[0].shape)

            # augment inputs with noise
            noise = torch.randn_like(inputs, device='cuda') * noise_sd

            if args.adv_training:
                requires_grad_(model, False)
                model.eval()
                inputs = attacker.attack(model, inputs, targets, 
                                        noise=noise, 
                                        num_noise_vectors=args.num_noise_vec, 
                                        no_grad=args.no_grad_attack
                                        )
                model.train()
                requires_grad_(model, True)
            
            if args.train_multi_noise:
                noisy_inputs = inputs + noise

                targets = targets.unsqueeze(1).repeat(1, args.num_noise_vec).reshape(-1,1).squeeze()
                outputs = model(noisy_inputs)
                loss = criterion(outputs, targets)

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), noisy_inputs.size(0))
                top1.update(acc1.item(), noisy_inputs.size(0))
                top5.update(acc5.item(), noisy_inputs.size(0))
                    
                # compute gradient and do SGD step
                optimizer.zero_grad()
                if dataLoaderIdx == PSEUDO_LABELLED:
                    loss *= args.self_training_weight
                loss.backward()
                optimizer.step()
            
            else:
                inputs = inputs[::args.num_noise_vec] # subsample the samples
                noise = noise[::args.num_noise_vec]
                # noise = torch.randn_like(inputs, device='cuda') * noise_sd
                noisy_inputs_list.append(inputs + noise)

        if not args.train_multi_noise:
            noisy_inputs = torch.cat(noisy_inputs_list)
            targets = batch[1].cuda()
            assert len(targets) == len(noisy_inputs)

            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), noisy_inputs.size(0))
            top1.update(acc1.item(), noisy_inputs.size(0))
            top5.update(acc5.item(), noisy_inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if dataLoaderIdx == PSEUDO_LABELLED:
                loss *= args.self_training_weight
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float, attacker: Attacker=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_normal = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()
    requires_grad_(model, False)


    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            noise = torch.randn_like(inputs, device='cuda') * noise_sd
            noisy_inputs = inputs + noise
            
            # compute output
            if args.adv_training:
                normal_outputs = model(noisy_inputs)
                acc1_normal, _ = accuracy(normal_outputs, targets, topk=(1, 5))
                top1_normal.update(acc1_normal.item(), inputs.size(0))

                with torch.enable_grad():
                    inputs = attacker.attack(model, inputs, targets, noise=noise)
                # noise = torch.randn_like(inputs, device='cuda') * noise_sd
                noisy_inputs = inputs + noise

            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        if args.adv_training:
            return (losses.avg, top1.avg, top1_normal.avg)
        else:
            return (losses.avg, top1.avg, None)


if __name__ == "__main__":
    main()
