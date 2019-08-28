import bisect 
import os
import pickle

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity
from typing import *
from zipdata import ZipData


# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
IMAGENET_ON_PHILLY_DIR = "/hdfs/public/imagenet/2012/"

# list of all datasets
DATASETS = ["imagenet", "imagenet32", "cifar10"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        if "PT_DATA_DIR" in os.environ: #running on Philly
            return _imagenet_on_philly(split)
        else:
            return _imagenet(split)

    elif dataset == "imagenet32":
        return _imagenet32(split)
    
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
    elif dataset == "imagenet32":
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
    dataset_path = os.path.join('datasets', 'dataset_cache')
    if split == "train":
        return datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.ToTensor())

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

def _imagenet32(split: str) -> Dataset:
    dataset_path = os.path.join('datasets', 'Imagenet32')
   
    if split == "train":
        return ImageNetDS(dataset_path, 32, train=True, transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor()]))
    
    elif split == "test":
        return ImageNetDS(dataset_path, 32, train=False, transform=transforms.ToTensor())

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


# from https://github.com/hendrycks/pre-training
class ImageNetDS(Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.

    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


## To use this dataset, please contact the authors of https://arxiv.org/pdf/1905.13736.pdf
# to get access to this pickle file (ti_top_50000_pred_v3.1.pickle) containing the dataset.
class TiTop50KDataset(Dataset):
            """500K images closest to the CIFAR-10 dataset from 
               the 80 Millon Tiny Images Datasets"""
            def __init__(self):
                super(TiTop50KDataset, self).__init__()
                dataset_path = os.path.join('datasets', 'ti_top_50000_pred_v3.1.pickle')

                self.dataset_dict = pickle.load(open(dataset_path,'rb'))
                #{'data', 'extrapolated_targets', 'ti_index', 
                # 'prediction_model', 'prediction_model_epoch'}
                
                self.length = len(self.dataset_dict['data'])
                self.transforms = transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()
                        ])

            def __getitem__(self, index):
                img = self.dataset_dict['data'][index]
                target = self.dataset_dict['extrapolated_targets'][index]
                
                img = Image.fromarray(img)
                img = self.transforms(img)
        
                return img, target
                        
            def __len__(self):
                return self.length


class MultiDatasetsDataLoader(object):
    """Dataloader to alternate between batches from multiple dataloaders 
    """
    def __init__(self, task_data_loaders, equal_num_batch=True, start_iteration=0):
        if equal_num_batch:
            lengths = [len(task_data_loaders[0]) for i,_ in enumerate(task_data_loaders)]
        else:
            lengths = [len(data_loader) for data_loader in task_data_loaders]
    
        self.task_data_loaders = task_data_loaders
        self.start_iteration = start_iteration
        self.length = sum(lengths)
        self.dataloader_indices = np.hstack([
            np.full(task_length, loader_id)
            for loader_id, task_length in enumerate(lengths)
        ])

    def __iter__(self):
        self.task_data_iters = [iter(data_loader)
                                for data_loader in self.task_data_loaders]
        self.cur_idx = self.start_iteration
        # synchronizing the task sequence on each of the worker processes
        # for distributed training. The data will still be different, but
        # will come from the same task on each GPU.
        # np.random.seed(22)
        np.random.shuffle(self.dataloader_indices)
        # np.random.seed()
        return self

    def __next__(self):
        if self.cur_idx == len(self.dataloader_indices):
            raise StopIteration
        loader_id = self.dataloader_indices[self.cur_idx]
        self.cur_idx += 1
        return next(self.task_data_iters[loader_id]), loader_id

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return self.length

    @property
    def num_tasks(self):
        return len(self.task_data_iters)
