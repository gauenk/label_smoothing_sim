"""
Generate Similarity and KNN matrices from 
neural networks trained on the CIFAR dataset
"""


# python imports
import uuid,gc,sys
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
from datasets.cifar import *
from mnist_knn_parser import parse_args,set_cfg_from_args # still mnist parser
from nn_knn.utils import train_step,test_step,save_results
from learning.train import thtrain_cls
from learning.test import thtest_cls
#from base.timer import Timer
from base.config import save_cfg,read_cfg_file,get_cfg_checkpoint_fn
from base.thutils import load_pytorch_model,PytorchModelWrapper,FeatureExtractor,save_pytorch_model
from base.utils import *


def load_cifar_model(cfg):

    # load pytorch model
    model_class = load_pytorch_model(cfg.model_py,cfg.model_th)
    th_model = model_class().to(cfg.device)
    model = PytorchModelWrapper(th_model)

    # load feature extractor; e.g. "outputs" is more than final layer
    layernames = [th_model.ftr_name,th_model.output_name]
    ftrModel = FeatureExtractor(model,layernames)

    return ftrModel

def main(args = None):
    print("Generating data for knn with cifar data")

    # --------------
    # Load settings
    # --------------

    cfg = load_cifar_cfg()
    if args is None: args = parse_args() # enable multiprocess
    set_cfg_from_args(cfg,args)
    if args.cfg_file: cfg = read_cfg_file(args.cfg_file) # overwrite with cfg 
    device = cfg.device
    cfg.use_cuda = args.no_cuda and torch.cuda.is_available()
    print("Experiment Name: {}".format(cfg.exp_name))
    save_cfg(cfg)
    
    # -------------------
    # Load model & data
    # -------------------

    # load the cifar data
    data = edict()
    data.tr,data.te,data.tr_sub = load_cifar_data(cfg)

    # load the cifar model
    model = load_cifar_model(cfg)
    th_model = model.th_model
    path = get_cfg_checkpoint_fn(cfg).format(0)
    save_pytorch_model(model.th_model,path)

    # ----------------
    # Run Experiments
    # ----------------

    # inputs for train/test functions
    optimizer = optim.Adadelta(model.th_model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.gamma)
    inputs = [cfg, model, device, optimizer]

    # te = test_step(1,data.te,*inputs[:-1])
    for epoch in range(cfg.epochs):

        # train & test the model
        print("Training step")
        tr = train_step(epoch,data.tr,data.tr_sub,*inputs)
        print("-=-=-=-=- Testing step -=-=-=-=-")
        te = test_step(epoch,data.te,*inputs[:-1])
        save_results(tr,te,epoch,cfg)
        # thtest_cls(cfg, th_model, device, data.te)

        gc.collect()
        

    print(f"Finished running experiment {cfg.exp_name}")

if __name__ == "__main__":
    main()

