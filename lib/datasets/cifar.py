"""
File for functions handling cifar data

"""

# python imports
import importlib.util
import os,warnings,uuid
from functools import partial
import os.path as osp
import numpy as np
import numpy.random as npr
from PIL import Image
from easydict import EasyDict as edict

# torch imports
import torch as th
import torch.nn as nn
from torchvision import datasets,transforms

# project imports
import settings
from base.thutils import apply_label_noise_level,get_train_subset_indices
from label_denoising.configs import get_cfg_graphKernel_2006

class noisyCIFAR(datasets.CIFAR10):
    """
    This wrapper just rewrites the original CIFAR class 
    overwriting the __getitem__ to make target_transform
    accept two inputs: target and index.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        print("ROOT:", root)
        super(noisyCIFAR, self).__init__( root, train=train, transform=transform,
                                          target_transform=target_transform,
                                          download=download)
        print(self.transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target,index)

        return img, target


def vanilla_cifar_trdata(cfg):
    data_loader = th.utils.data.DataLoader(
        noisyCIFAR(f'{cfg.root_path}/data/CIFAR10/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=1, shuffle=False)
    return data_loader

def load_cifar_data(cfg):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
    
    # create label noise transform
    apply_label_noise = None
    if cfg.label_noise is not None and cfg.label_noise > 0:
        ds = vanilla_cifar_trdata(cfg)
        targets = ds.dataset.targets
        apply_label_noise = apply_label_noise_level(cfg,targets)

    # train loader
    train_loader = th.utils.data.DataLoader(
        noisyCIFAR(f'{cfg.root_path}/data/CIFAR10/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),
                       target_transform=apply_label_noise),
        batch_size=cfg.batch_size, shuffle=True, **kwargs)

    # subset train loader for fast checkpoint evaluation
    train_subset_indices = get_train_subset_indices(train_loader)
    train_subset_loader = th.utils.data.DataLoader(
        th.utils.data.Subset(
            noisyCIFAR(f'{cfg.root_path}/data/CIFAR10/', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]),
                           target_transform=apply_label_noise),
            train_subset_indices
            ),
        batch_size=cfg.batch_size, shuffle=False, **kwargs)

    # test loader
    test_loader = th.utils.data.DataLoader(
        datasets.CIFAR10(f'{cfg.root_path}/data/CIFAR10/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=cfg.test_batch_size, shuffle=False, **kwargs)

    return train_loader,test_loader,train_subset_loader

def load_cifar_cfg():
    cfg = edict()

    cfg.exp_name = uuid.uuid4()
    cfg.output_dir = 'cifar_knn'

    cfg.device = 'cuda'
    cfg.use_cuda = True

    cfg.model_py = "./models/cifar10/graphconv_default.py"
    cfg.ftr_layer_name = 'h_output'
    # cfg.model_py = "./models/cifar10/pytorch_default.py"
    # cfg.ftr_layer_name = 'fc1'
    cfg.model_th = None

    cfg.batch_size = 128
    cfg.gamma = 0.7
    cfg.log_interval = 10
    cfg.lr = 1.0
    cfg.seed = 1
    cfg.test_batch_size = 1000
    cfg.epochs = 100
    cfg.save_epochs = [] 

    cfg.nclasses = 10
    cfg.n_train_samples = 50000
    cfg.n_test_samples = 10000
    cfg.root_path = settings.ROOT_PATH

    cfg.label_noise = .25
    cfg.smoothing = True
    cfg.graphKernel2006 = get_cfg_graphKernel_2006(alpha=0.1)

    return cfg

def load_cifar_model(cfg):

    # load pytorch model
    model_class = load_pytorch_model(cfg.model_py,cfg.model_th)
    th_model = model_class().to(cfg.device)
    model = PytorchModelWrapper(th_model)

    # load feature extractor; e.g. "outputs" is more than final layer
    layernames = [th_model.ftr_name,th_model.output_name]
    ftrModel = FeatureExtractor(model,layernames)

    return ftrModel



