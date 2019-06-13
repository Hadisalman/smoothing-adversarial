import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from typing import *
from zipdata import ZipData


# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
IMAGENET_ON_PHILLY_DIR = "/hdfs/public/imagenet/2012/"

# list of all datasets
DATASETS = ["imagenet", "cifar10"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        if "PT_DATA_DIR" in os.environ: #running on Philly
            return _imagenet_on_philly(split)
        else:
            return _imagenet(split)
    
    elif dataset == "cifar10":
        return _cifar10(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    if dataset == "imagenet":
        return InputCenterLayer(_IMAGENET_MEAN)
    elif dataset == "cifar10":
        return InputCenterLayer(_CIFAR10_MEAN)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10('dataset_cache', train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10('dataset_cache', train=False, download=True, transform=transforms.ToTensor())

    elif split in ["mini_labelled", "mini_unlabelled", "mini_test"]:
        return HybridCifarDataset(split)
        # return MiniCifarDataset(split)

    else:
        raise Exception("Unknown split name.")

def _imagenet_on_philly(split: str) -> Dataset:
        
        trainpath = os.path.join(IMAGENET_ON_PHILLY_DIR, 'train.zip')
        train_map = os.path.join(IMAGENET_ON_PHILLY_DIR, 'train_map.txt')
        valpath = os.path.join(IMAGENET_ON_PHILLY_DIR, 'val.zip')
        val_map = os.path.join(IMAGENET_ON_PHILLY_DIR, 'val_map.txt')

        if split == "train":
            return ZipData(trainpath, train_map,
                            transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            ]))
        elif split == "test":
            return ZipData(valpath, val_map, 
                            transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            ]))

def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds


class InputCenterLayer(torch.nn.Module):
    """Centers the channels of a batch of images by subtracting the dataset mean.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(InputCenterLayer, self).__init__()
        self.means = torch.tensor(means).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return input - means
