import os
import shutil

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def copy_code(outdir):
    """Copies files to the outdir to store complete script with each experiment"""
    # embed()
    code = []
    exclude = set([])
    for root, _, files in os.walk("./code", topdown=True):
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]

    for r, f in code:
        codedir = os.path.join(outdir,r)
        if not os.path.exists(codedir):
            os.mkdir(codedir)
        shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
    print("Code copied to '{}'".format(outdir))