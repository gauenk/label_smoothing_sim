import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import _init_paths
from easydict import EasyDict as edict
from block_gen import block_gen as bg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sim_knn import sim_knn as sim_knn
from mpl_toolkits.mplot3d import axes3d, Axes3D 

def experiment_setup(cfg):
    exp_info = edict()


    # step 1: generate grid
    base_grid,base_l,base_gt = bg.generate_blockgrid(cfbg.size,cfbg.nclasses,
                                                     random=cfbg.random,
                                                     bg_grid_noise=cfbg.bg_grid_noise,
                                                     label_noise=cfbg.label_noise)
    exp_info.base = edict()
    exp_info.base.grid = base_grid
    exp_info.base.labels = base_labels
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
    knn_samples = sim_knn.pseudo_knn(grid,cfk.target,cfk.k)
    sorted_knn_samples = bg.value_from_index(order_idx,knn_samples)
    exp_info.knn = edict()
    exp_info.knn.samples = knn_samples
    exp_info.knn.sorted_samples = sorted_knn_samples

    return exp_info
