import numpy as np
import numpy.random as npr
from sim_knn import sim_knn as sim_knn


def maximum_vote(grid,labels):
    m = grid.shape[0]
    classes = np.unique(labels)
    nclasses = len(classes)
    cls_scores = np.zeros( (m,nclasses) )
    for cls in range(nclasses):
        cls_idx = np.where(labels == classes[cls])[0]
        cls_scores[:,cls] = np.sum(grid[:,cls_idx],axis=1)
    cls_scores = cls_scores / np.sum(cls_scores,axis=0)
    preds = np.argmax(cls_scores,axis=1)[:,np.newaxis]
    return preds

def maximum_vote_knn(grid,labels,knn_k):
    nrows,ncols = grid.shape[0:2]
    knn_samples = np.zeros( (nrows, knn_k) , dtype=np.int)
    guess = np.zeros( (nrows,1), dtype=np.int)
    for i in range(nrows):
        knn_samples[i,:] = sim_knn.pseudo_knn(grid,i,knn_k)
        #knn_labels[i,:] = labels[knn_samples[i,:]].T # pick the correct k
        knn_labels = np.squeeze(labels[knn_samples[i,:]]).astype(np.int)
        #print(knn_labels,np.bincount(knn_labels),np.argmax(np.bincount(knn_labels)))
        guess[i] = np.argmax(np.bincount(knn_labels))
    return guess

def maximum_vote_grid(exp_data,cfg):
    """
    use maximum vote for label denoising
    uses the *entire grid*
    e.g. using knn with the entire dataset
    """
    grid,labels = exp_data.base.grid,exp_data.base.labels
    return maximum_vote(grid,labels)

def maximum_vote_subgrid(exp_data,cfg):
    """
    use maximum vote for label denoising
    """
    subgrid,slabels = exp_data.subset.grid,exp_data.subset.labels
    return maximum_vote(grid,labels)

def maximum_vote_grid_knn(exp_data,cfg):
    """
    use maximum vote for label denoising
    """
    grid = exp_data.base.grid
    labels = exp_data.base.labels
    if cfg.base.grid_transform is not None:
        grid = cfg.base.grid_transform(grid)
    return maximum_vote_knn(grid,labels,cfg.knn.k)

def maximum_vote_subgrid_knn(exp_data,cfg):
    """
    use maximum vote for label denoising
    """
    grid = exp_data.subset.grid
    labels = exp_data.subset.labels
    if cfg.subset.grid_transform is not None:
        grid = cfg.base.grid_transform(grid)
    return maximum_vote_knn(grid,labels,cfg.knn.k)

    # grid = exp_data.subset.grid
    # labels = exp_data.subset.labels



