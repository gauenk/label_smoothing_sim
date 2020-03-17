import copy
import numpy as np
import numpy.random as npr
from easydict import EasyDict as edict
import matplotlib.pyplot as plt


# project imports!!
from sim_knn import sim_knn as sim_knn
from block_gen import block_gen as bg


def experiment_setup(cfg):
    """
    Generate experiment data from config edict
    """
    cfbg = cfg.base
    cfk = cfg.knn
    cfs = cfg.subset

    exp_info = edict()

    # step 1: generate grid
    base_grid,base_l,base_gt = bg.generate_blockgrid(cfbg.size,cfbg.nclasses,
                                                     random=cfbg.random,
                                                     bg_grid_noise=cfbg.bg_grid_noise,
                                                     label_noise=cfbg.label_noise)
    exp_info.base = edict()
    exp_info.base.grid = base_grid
    exp_info.base.labels = base_l
    exp_info.base.gt = base_gt

    # step 2: subsample grid
    grid,subset_idx = bg.subsample_grid(base_grid,cfs.method,sort=cfs.sort)
    l = base_l[subset_idx]
    gt = base_gt[subset_idx]
    exp_info.subset = edict()
    exp_info.subset.grid = grid
    exp_info.subset.labels = l
    exp_info.subset.gt = gt

    # step 3: computed ordered subgrid information
    order_idx = np.argsort(subset_idx)
    sorted_grid = bg.reoder_grid_by_rowvec(grid,order_idx)
    sorted_l = l[order_idx]
    sorted_gt = gt[order_idx]
    exp_info.subset.ordering = order_idx
    exp_info.subset.sorted_grid = sorted_grid
    exp_info.subset.sorted_l = sorted_l
    exp_info.subset.sorted_gt = sorted_gt

    # step 3: simulate knn
    # knn_samples = sim_knn.pseudo_knn(grid,cfk.target,cfk.k)
    # sorted_knn_samples = bg.value_from_index(order_idx,knn_samples)
    # exp_info.knn = edict()
    # exp_info.knn.samples = knn_samples
    # exp_info.knn.sorted_samples = sorted_knn_samples

    return exp_info


def create_label_noise_experiment_cfg(base_cfg,label_noise_levels):
    cfg_list = []
    for label_noise in label_noise_levels:
        exp_cfg = copy.deepcopy(base_cfg)
        exp_cfg.base.label_noise = label_noise
        cfg_list.append(exp_cfg)
    return cfg_list

def create_knn_k_experiment_cfg(base_cfg,knn_k_list):
    cfg_list = []
    for knn_k in knn_k_list:
        exp_cfg = copy.deepcopy(base_cfg)
        exp_cfg.knn.k = int(knn_k)
        cfg_list.append(exp_cfg)
    return cfg_list


def run_experiments(cfg_list,smoothing_function,nrepeats=300):
    nexperiments = len(cfg_list)

    # results to return
    results = edict()
    results.acc = np.zeros( (nexperiments, nrepeats) )

    for i in range(nexperiments):
        for r in range(nrepeats):
            exp_cfg = cfg_list[i]

            # experiment setup
            exp_data = experiment_setup(exp_cfg)
            gt = exp_data.base.gt

            # simulate experiment
            guess = smoothing_function(exp_data,exp_cfg)

            # save results
            acc = 100 * np.sum(guess == gt)/len(gt)
            results.acc[i,r] = acc
    return results

def plot_results_field(results_field,xticklabels=None,xaxis_label=None,ax=None):
    nexperiments = results_field.shape[0]
    x = range(nexperiments)
    mean = np.mean(results_field,axis=1)
    std = np.std(results_field,axis=1)
    y = mean
    error = std / np.sqrt(len(y))

    fig = None
    if ax is None:
        fig,ax = plt.subplots()
    handle = ax.plot(x,y)
    ax.fill_between(x, y-error, y+error, alpha=0.5)
    ax.set_ylabel("Accuracy",fontsize=15)

    if xaxis_label:
        ax.set_xlabel(xaxis_label,fontsize=15)
    if xticklabels is not None:
        nlabels = len(xticklabels)
        ax.set_xticks(np.linspace(0,nexperiments,nlabels))
        ax.set_xticklabels(xticklabels)

    return fig,ax,handle
