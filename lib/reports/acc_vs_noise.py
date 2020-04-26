"""
This file contains function for plotting 
accuracy versus noise level
using a set of 
"experiment_uuids"
as the input.
"""

# python imports
import os,sys
import numpy as np
from easydict import EasyDict as edict

# project imports
import settings
from base import plot as localplot
from .utils import *


def main(exp_uuids,root_dir,cls_info_fn):
    configs = load_configs(exp_uuids,root_dir)
    cls_infos_tr,cls_infos_te = load_cls_infos(exp_uuids,root_dir,cls_info_fn)
    # plot_set_results(cls_infos_tr,configs,"tr")
    # plot_set_results(cls_infos_te,configs,"te")

    plot_set_results_smoothing(cls_infos_tr,configs,"tr")
    plot_set_results_smoothing(cls_infos_te,configs,"te")

def plot_set_results(cls_infos_set,configs,setStr): # setStr :: "tr" = "train & "te" = "test"
    acc,prec,rec,noise = gatherNumpys(cls_infos_set,configs)
    plot_acc_v_noise(acc,noise,'no correction',setStr)

def plot_acc_v_noise(acc,noise,name,setStr,ax=None):

    order = np.argsort(noise)
    noise = noise[order]
    acc = acc[order]

    if ax is not None:
        ax.plot(noise,acc[:,-1],'-x',label=name)
    else:
        fig,ax = plt.subplots(1,figsize=(8,6))
        ax.set_title("GraphKernel 2006",fontsize=12)
        ax.set_ylabel("Accuracy",fontsize=12)
        ax.set_xlabel("Noise Level (% randomly corrupted)",fontsize=12)
        legend_str = [name]
        localplot.add_legend(ax,"Methods",legend_str,shrink=True,fontsize=12,framealpha=1.0)
        plot_fn = 'noise_v_acc_{}.png'.format(setStr)
        plt.savefig(plot_fn,transparent=False)
        # localplot.add_legend(ax,"Methods",legend_str,shrink=True,fontsize=12,framealpha=0.0)
        # plt.savefig('noise_v_acc_tr.png',transparent=True)

def plot_set_results_smoothing(cls_infos_set,configs,setStr):
    
    results,noise,alpha = gatherNumpysWithSmoothing(cls_infos_set,configs)

    fig,ax = plt.subplots(1,figsize=(8,6))
    ualphas = np.unique(alpha)
    names = []
    for fixed_alpha in ualphas:
        indices = np.where(alpha == fixed_alpha)
        alpha_noise = noise[indices]
        alpha_acc = results.sm_acc[indices] # smoothed only
        alpha_str = '{:.2}'.format(fixed_alpha)
        plot_acc_v_noise(alpha_acc,alpha_noise,alpha_str,setStr,ax=ax)
        names.append(alpha_str)
    ax.set_title("Baseline Methods",fontsize=12)
    ax.set_ylabel("Accuracy",fontsize=12)
    ax.set_xlabel("Noise Level (% randomly corrupted)",fontsize=12)
    localplot.add_legend(ax,"Methods",names,shrink=True,fontsize=12,framealpha=1.0)
    plot_dir = osp.join(settings.ROOT_PATH,'output/acc_v_noise')
    if not osp.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_fn = osp.join(plot_dir,'noise_v_acc_alphas_{}.png'.format(setStr))
    plt.savefig(plot_fn,transparent=False)

def gatherNumpysWithSmoothing(cls_infos,configs):
    acc = []
    prec = []
    rec = []
    
    sm_acc = []
    sm_prec = []
    sm_rec = []

    noise = []
    alpha = []
    for exp_id in cls_infos.keys():
        cfg = configs[exp_id]
        noise.append(cfg.label_noise)
        alpha.append(cfg.graphKernel2006.alpha)
        acc.append([]),prec.append([]),rec.append([])
        sm_acc.append([]),sm_prec.append([]),sm_rec.append([])
        print(exp_id)
        for index,results in cls_infos[exp_id].items():
            acc[-1].append(results.acc)
            prec[-1].append(results.precision)
            rec[-1].append(results.recall)

            if 'sm_acc' in results.keys():
                sm_acc[-1].append(results.sm_acc)
                sm_prec[-1].append(results.sm_precision)
                sm_rec[-1].append(results.sm_recall)

    noise = np.array(noise) # Num of Exps
    alpha = np.array(alpha) # Num of Exps

    acc = np.array(acc) # Num of Exp \times Num of Epochs
    prec = np.array(prec) # Num of Exp \times Num of Epochs \times Num of Classes
    rec = np.array(rec) # Num of Exp \times Num of Epochs \times Num of Classes
    
    sm_acc = np.array(sm_acc)
    sm_prec = np.array(sm_prec)
    sm_rec = np.array(sm_rec)


    results = {'acc':acc,'prec':prec,'rec':rec,
               'sm_acc':sm_acc,'sm_prec':sm_prec,'sm_rec':sm_rec}
    results = edict(results)
    return results,noise,alpha

def gatherNumpys(cls_infos,configs):
    acc = []
    prec = []
    rec = []
    
    noise = []
    for exp_id in cls_infos.keys():
        cfg = configs[exp_id]
        noise.append(cfg.label_noise)
        acc.append([]),prec.append([]),rec.append([])
        for index,results in cls_infos[exp_id].items():
            acc[-1].append(results.acc)
            prec[-1].append(results.precision)
            rec[-1].append(results.recall)

    noise = np.array(noise) # Num of Exps

    acc = np.array(acc) # Num of Exp \times Num of Epochs
    prec = np.array(prec) # Num of Exp \times Num of Epochs \times Num of Classes
    rec = np.array(rec) # Num of Exp \times Num of Epochs \times Num of Classes

    return acc,prec,rec,noise
