"""
Common functions used for running DATASET_knn.py files
Example: tools/mnist_knn.py

This script is "down the import tree";
e.g. We have

base/* -> cache/* -> other packages -> nn_knn/utils.py

"""

# python imports
import os,uuid
import os.path as osp
from easydict import EasyDict as edict
import numpy as np
from scipy.spatial.distance import pdist

# project imports (down the import tree)
import settings
from learning.train import thtrain_cls
from base.timer import Timer
from base.thutils import *
from base.utils import *
from label_denoising.graph_kernel_2006 import graphKernel2006
from base.config import get_cfg_checkpoint_fn,get_cfg_checkpoint_dir

def save_results(tr,te,epoch,cfg):
    checkpoint_dir = get_cfg_checkpoint_dir(cfg)
    rdir = osp.join(checkpoint_dir, "results_{}/".format(epoch))
    if not osp.exists(rdir):
        os.makedirs(rdir)

    # setup path skeletons
    cls_info_path = osp.join(rdir, "cls_info_{}.pkl")
    feature_path = osp.join(rdir, "features_{}.npy")
    sim_mat_path = osp.join(rdir, "sim_mat_{}.npy")

    # save cls info
    tr_cls = {'acc':tr.acc,'recall':tr.recall,'precision':tr.precision,
              'sm_acc':tr.sm_acc,'sm_recall':tr.sm_recall,'sm_precision':tr.sm_precision,
    }
    te_cls = {'acc':te.acc,'recall':te.recall,'precision':te.precision,
              'sm_acc':te.sm_acc,'sm_recall':te.sm_recall,'sm_precision':te.sm_precision,
    }
    write_pickle(tr_cls,cls_info_path.format("tr"))
    write_pickle(te_cls,cls_info_path.format("te"))

    # save features
    np.save(feature_path.format("tr"),tr.features)
    np.save(feature_path.format("te"),te.features)
    
    # save sim_mat
    # np.save(sim_mat_path.format("tr"),tr.sim_mat)
    # np.save(sim_mat_path.format("te"),te.sim_mat)
    
def init_result_dict(epochs):
    t = edict()
    t.acc,t.recall,t.precision = [],[],[]
    t.ap,t.mAP = [],[]
    t.features,t.sim_mat = [],[]
    return t
    
def load_cls_info(t,path):
    data = read_pickle(path)
    t.acc.append(data['acc'])
    t.recall.append(data['recall'])
    t.precision.append(data['precision'])

def load_features(t,path):
    features = np.load(path)
    t.features.append(features)

def load_simmat(t,path):
    sim_mat = np.load(path)
    t.sim_mat.append(sim_mat)

def load_results(cfg, _load_cls_ = True, _load_ftrs_ = True, _load_simmat_ = True):
    exp_dir = osp.join(cfg.root_path,f"output/{cfg.output_dir}/{cfg.exp_name}")
    print(exp_dir)
    tr = init_result_dict(cfg.epochs)
    te = init_result_dict(cfg.epochs)
    for epoch in range(cfg.epochs):
        results_dir = osp.join(exp_dir,f'results_{epoch}')

        # cls info
        if _load_cls_:
            clsinfo_fn_tr = osp.join(results_dir,'cls_info_tr.pkl')
            load_cls_info(tr,clsinfo_fn_tr)
            clsinfo_fn_te = osp.join(results_dir,'cls_info_te.pkl')
            load_cls_info(te,clsinfo_fn_te)

        # features
        if _load_ftrs_:
            features_fn_tr = osp.join(results_dir,'features_tr.npy')
            load_features(tr,features_fn_tr)
            features_fn_te = osp.join(results_dir,'features_te.npy')
            load_features(te,features_fn_te)

        # sim mats
        if _load_simmat_:
            simmat_fn_tr = osp.join(results_dir,'sim_mat_tr.npy')
            load_simmat(tr,simmat_fn_tr)
            simmat_fn_te = osp.join(results_dir,'sim_mat_te.npy')
            load_simmat(te,simmat_fn_te)
    return tr,te


def get_confmat_from_raw(outputs,data_loader):
    ncls = 10
    cls_sum = np.zeros(ncls,dtype=np.int)
    conf_mat = np.zeros((ncls,ncls),dtype=np.int)
    prev = 0
    for data, target in data_loader:
        start = prev
        end = start + len(target)
        guesses = np.argmax(outputs['cls'][start:end],axis=1)
        prev = end
        for guess,cls in zip(guesses,target): # unravel the batch
            conf_mat[guess,cls] += 1
    return conf_mat

def get_smoothed_confmat_from_raw(cfg,outputs,data_loader):
    ncls = 10
    cls_sum = np.zeros(ncls,dtype=np.int)
    conf_mat = np.zeros((ncls,ncls),dtype=np.int)
    prev = 0
    for data, target in data_loader:
        start = prev
        end = start + len(target)
        guesses = graphKernel2006(cfg.graphKernel2006,
                                  outputs['cls'][start:end],
                                  outputs[cfg.ftr_layer_name][start:end])
        prev = end
        for guess,cls in zip(guesses,target): # unravel the batch
            conf_mat[guess,cls] += 1
    return conf_mat

def get_acc_from_confmat(confmat):
    return np_divide( np.trace(confmat), np.sum(confmat) )

def get_recall_from_confmat(confmat):
    # sum cols; total number of labels per class
    ncls = np.sum(confmat,axis=1)
    return np_divide( np.diag(confmat), ncls )

def get_precision_from_confmat(confmat):
    # sum rows; total number of guesses per class
    ngss = np.sum(confmat,axis=0)
    return np_divide(np.diag(confmat), ngss)

def get_ap_from_raw(outputs,data):
    pass

def get_map_from_ap(outputs,data):
    pass

def get_simmat_from_features(X):
    print(X.shape)
    s = 1.
    pairwise_dists = pdist(X, 'sqeuclidean')
    K = np.exp( - pairwise_dists ** 2 / s ** 2)
    return K

def train_step(epoch, data, subset, cfg, model, device, optimizer):
    # train
    thtrain_cls(cfg, model.th_model, device, data, optimizer, epoch)

    # save updated params
    if epoch in cfg.save_epochs or cfg.epochs == epoch + 1:
        path = get_cfg_checkpoint_fn(cfg).format(epoch+1)
        save_pytorch_model(model.th_model,path)

    # compute train results
    return test_step(model, subset, cfg, model, device)

def test_step(epoch, data, cfg, model, device):

    # get raw output; this can be for several layers
    outputs = pytorch_model_outputs(model,data,device)
    
    # cls metrics
    confmat = get_confmat_from_raw(outputs,data)
    print(confmat)
    acc = get_acc_from_confmat(confmat)
    recall = get_recall_from_confmat(confmat)
    precision = get_precision_from_confmat(confmat)

    if cfg.smoothing:
        sm_confmat = get_smoothed_confmat_from_raw(cfg,outputs,data)
        print(sm_confmat)
        sm_acc = get_acc_from_confmat(sm_confmat)
        sm_recall = get_recall_from_confmat(sm_confmat)
        sm_precision = get_precision_from_confmat(sm_confmat)
    # ap = get_ap_from_raw(outputs,data)
    # mAP = get_map_from_ap(ap,data)

    # similarity metric
    features = get_features_from_raw(outputs)
    print("simmat")
    t = Timer()
    t.tic()
    sim_mat = get_simmat_from_features(features)
    print(sim_mat.shape)
    t.toc()
    print(t)
    print("end [simmat]")

    # store results
    results = edict()
    results.acc = acc
    results.recall = recall
    results.precision = precision
    if cfg.smoothing:
        results.sm_acc = sm_acc
        results.sm_recall = sm_recall
        results.sm_precision = sm_precision
    # results.ap = ap
    # results.mAP = mAP
    print(results)
    results.features = features
    results.sim_mat = sim_mat

    return results


