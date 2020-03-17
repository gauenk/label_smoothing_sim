"""
Generate Similarity and KNN matrices from 
neural networks trained on the MNIST dataset
"""


# python imports
import uuid
from easydict import EasyDict as edict

# pytorch imports
import torch
import torch.nn.functional as F

# project imports
from mnist_knn_parser import parser_args,set_cfg_from_args
from learning.train import pytrain_cls
from learning.test import pytest_cls
from base.pytorch import *


def load_cfg():
    cfg = edict()
    cfg.epochs = 10
    cfg.device = 'cuda'
    cfg.exp_name = 'testing'
    cfg.use_cuda = False
    cfg.model_py = "./models/mnist/pytorch_default.py"
    cfg.model_th = None
    #cfg.exp_name = uuid.uuid4()

def load_mnist_model(cfg):

    # load pytorch model
    model_class = load_pytorch_model(cfg.model_py,cfg.model_th)
    model = model_class().to(cfg.device)
    model = PytorchModelWrapper(model)

    # load feature extractor; e.g. "outputs" is more than final layer
    layernames = [model.ftr_name,model.output_name]
    ftrModel = FeatureExtractor(model,layernames)

    return ftrmodel

def load_mnist_data(cfg):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=cfg.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=cfg.batch_size, shuffle=True, **kwargs)
    return train_loader,test_loader

def train_step(epoch, data, args, model, device, optimizer):
    # train
    pytrain_cls(args, model.py_model, device, data, optimizer, epoch)

    # save updated params
    save_model(model.py_model)

    # compute train results
    return test_step(model,data)


    acc = get_acc_from_raw(outputs,data)
    recall = get_recall_from_raw(outputs,data)
    precision = get_precision_from_raw(outputs,data)
    ap = get_ap_from_raw(outputs,data)
    mAP = get_map_from_ap(ap,data)

def get_acc_from_raw(outputs,data):
    pass

def get_recall_from_raw(outputs,data):
    pass

def get_precision_from_raw(outputs,data):
    pass

def get_ap_from_raw(outputs,data):
    pass

def get_map_from_ap(outputs,data):
    pass

def get_features_from_raw(outputs,data):
    pass

def get_simmat_from_raw(outputs,data):
    pass

def test_step(epoch, data, args, model, device):

    # get raw output; this can be for several layers
    outputs = pytorch_model_outputs(model,data)
    
    # cls metrics
    acc = get_acc_from_raw(outputs,data)
    recall = get_recall_from_raw(outputs,data)
    precision = get_precision_from_raw(outputs,data)
    ap = get_ap_from_raw(outputs,data)
    mAP = get_map_from_ap(ap,data)

    # similarity metric
    features = get_features_from_raw(outputs)
    sim_mat = get_simmat_from_raw(outputs)

    # store results
    results = edict()
    results.acc = acc
    results.recall = recall
    results.precision = precision
    results.ap = ap
    results.mAP = mAP
    results.ftrs = ftrs
    results.sim_mat = sim_mat

    return results


def main():

    # --------------
    # Load settings
    # --------------

    cfg = load_cfg()
    args = parse_args()
    set_cfg_from_args(cfg,args)
    device = cfg.device
    print(args.no_cuda,args.no-cuda)
    cfg.use_cuda = args.no_cuda and torch.cuda.is_available()
    
    # -------------------
    # Load model & data
    # -------------------

    # load the mnist model
    model = load_mnist_model(cfg)

    # load the mnist data
    data = edict()
    data.tr,data.te = load_mnist_data(cfg)

    # ----------------
    # Run Experiments
    # ----------------

    # inputs for train/test functions
    args = []
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    inputs = [args, model, device, optimizer]

    # gather data across several epochs of training
    results = edict() # classification results
    results.tr,results.te = [],[]
    for epoch in range(cfg.epochs):

        # train & test the model
        tr = train_step(epoch,data.tr,*inputs)
        te = test_step(epoch,data.te,*inputs[:-1])

        results.tr.append(tr)
        results.te.append(te)

    # ---------------------
    # Save Results & Plots
    # ---------------------

    # save the testing results
    save_results(results)
    
    
    

if __name__ == "__main__":
    print("HI")
    main()

