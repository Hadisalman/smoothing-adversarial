from archs.cifar_resnet import resnet as resnet_cifar
from datasets import get_normalize_layer, get_input_center_layer
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.models.resnet import resnet50


# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet110", "imagenet32_resnet110"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "imagenet32_resnet110":
        model = resnet_cifar(depth=110, num_classes=1000).cuda()

    # Both layers work fine, We tried both, and they both
    # give very similar results 
    # IF YOU USE ONE OF THESE FOR TRAINING, MAKE SURE
    # TO USE THE SAME WHEN CERTIFYING.
    normalize_layer = get_normalize_layer(dataset)
    # normalize_layer = get_input_center_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
