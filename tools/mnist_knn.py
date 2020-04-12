"""
Generate Similarity and KNN matrices from 
neural networks trained on the MNIST dataset
"""


# python imports
import uuid,gc
import os.path as osp
from easydict import EasyDict as edict
import numpy as np

# pytorch imports
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.optim.lr_scheduler import StepLR

# project imports
import _init_paths
from mnist_knn_parser import parse_args,set_cfg_from_args
from nn_knn.utils import *
from learning.train import thtrain_cls
from learning.test import thtest_cls
#from base.timer import Timer
from base.thutils import *
from base.utils import *

def load_mnist_model(cfg):

    # load pytorch model
    model_class = load_pytorch_model(cfg.model_py,cfg.model_th)
    th_model = model_class().to(cfg.device)
    model = PytorchModelWrapper(th_model)

    # load feature extractor; e.g. "outputs" is more than final layer
    layernames = [th_model.ftr_name,th_model.output_name]
    ftrModel = FeatureExtractor(model,layernames)

    return ftrModel

def main():
    print("Generating data for knn with mnist data")

    # --------------
    # Load settings
    # --------------

    cfg = load_mnist_cfg()
    args = parse_args()
    set_cfg_from_args(cfg,args)
    device = cfg.device
    cfg.use_cuda = args.no_cuda and torch.cuda.is_available()
    
    # -------------------
    # Load model & data
    # -------------------

    # load the mnist data
    data = edict()
    data.tr,data.te,data.tr_sub = load_mnist_data(cfg)

    # load the mnist model
    model = load_mnist_model(cfg)
    th_model = model.th_model
    path = cfg.checkpoint_fn.format(0)
    save_pytorch_model(model.th_model,path)

    # ----------------
    # Run Experiments
    # ----------------

    # inputs for train/test functions
    optimizer = optim.Adadelta(model.th_model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.gamma)
    inputs = [cfg, model, device, optimizer]

    for epoch in range(cfg.epochs):

        # train & test the model
        tr = train_step(epoch,data.tr,data.tr_sub,*inputs)
        te = test_step(epoch,data.te,*inputs[:-1])
        save_results(tr,te,epoch,cfg)
        thtest_cls(cfg, th_model, device, data.te)

        gc.collect()
        

    print(f"Finished running experiment {cfg.exp_name}")

if __name__ == "__main__":
    main()

