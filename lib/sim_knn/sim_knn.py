import numpy as np
import numpy.random as npr

def pseudo_knn(grid,i,k):
    m = grid.shape[0]
    weights = grid[i,:] / np.sum(grid[i,:])
    probs = weights**2 / sum(weights**2)

    nzi = probs.nonzero()[0]
    nz = len(nzi)
    if nz < k: # pick uniform from remaining dataset
        samples = np.zeros( k ).astype(np.int)
        samples[0:nz] = nzi
        choices = np.delete(np.arange(m-nz),nzi)
        indices = npr.choice(choices,size=k-nz,replace=False).astype(np.int)
        samples[nz:] = indices
    else:
        samples = npr.choice(m,size=k,replace=False,p=probs).astype(np.int)

    return samples


def pseudo_knn_mats(simmat,labels,knn_k):
    nrows,ncols = simmat.shape[0:2]
    knn_samples = np.zeros( (nrows, knn_k) , dtype=np.int )
    knn_labels = np.zeros( (nrows, knn_k), dtype=np.int )
    for i in range(nrows):
        knn_samples[i,:] = pseudo_knn(simmat,i,knn_k)
        knn_labels[i,:] = np.squeeze(labels[knn_samples[i,:]]).astype(np.int)
    return knn_samples,knn_labels
        
    
