"""
This script contains the functions required to generate blocks

"""

import numpy as np
import numpy.random as npr
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from base.utils import np_log

def generate_blockgrid(m,n, random=False, bg_grid_noise=0.5, label_noise = None):
    """
    m: dimension of the grid
    n: is the number of blocks
    bg_grid_noise: the magnitude of the noise in the background of the grid
    label_noise: percent of labels flipped
    """

    if n > m:
        raise ValueError("Number of blocks cannot exceed size of grid")

    if random:
        rvals = npr.rand( m**2 ).reshape(m,m)
        x = rvals * bg_grid_noise
    else:
        x = np.zeros( (m,m) )        

    y = np.zeros( (m,1) ) # group id
    gid = 0
    ridx = np.sort(npr.permutation(m-1)[0:n-1]+1)
    s = 0
    for r in ridx:
        if random:
            rvals = npr.rand( (r-s)**2 ).reshape(r-s,r-s)
            x[s:r,s:r] = rvals / 2. + 1/2.
        else:
            x[s:r,s:r] = 1
        y[s:r] = gid
        gid += 1
        s = r
    if random:
        rvals = npr.rand( (m-s)**2 ).reshape(m-s,m-s)
        x[s:,s:] = rvals / 2. + 1/2.
    else:
        x[s:,s:] = 1
    y[s:] = gid

    gt = np.copy(y)


    if label_noise: # [0,1]
        nflip = int(m * label_noise)
        flip_idx = npr.choice(m,size=nflip,replace=False) # which ones are flipped
        cls_probs = np.ones(n) / n
        new_labels = np.where(npr.multinomial(1,cls_probs,size=nflip) == 1)[1]
        new_labels = new_labels[:,np.newaxis]
        y[flip_idx] = new_labels

    return x,y,gt
    
def plot_blockgrid(*args):
    nargs = len(args)
    if nargs == 2:
        return plot_blockgrid_a(*args)
    elif nargs == 3:
        return plot_blockgrid_b(*args)
    elif nargs == 4:
        return plot_blockgrid_c(*args)
    else:
        raise ValueError(f"We can't accept [{nargs}] # args")

def plot_blockgrid_a(grid,labels):
    fig = plt.figure()
    gs = gridspec.GridSpec(1,2,width_ratios=[1,.2])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.imshow(grid)
    ax1.imshow(np.tile(labels,(1,10)))

    axs = [ax0,ax1]
    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axis('tight')

    return fig,axs

def plot_blockgrid_b(grid,labels,gt):
    fig = plt.figure()
    gs = gridspec.GridSpec(1,3,width_ratios=[1,.2,.2])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax0.imshow(grid)
    ax1.imshow(np.tile(labels,(1,10)))
    ax2.imshow(np.tile(gt,(1,10)))
    axs = [ax0,ax1,ax2]
    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axis('tight')

    return fig,axs

def plot_blockgrid_c(grid,labels,gt,rec):
    fig = plt.figure()
    gs = gridspec.GridSpec(1,4,width_ratios=[1,.15,.15,.15])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    ax0.imshow(grid)
    ax1.imshow(np.tile(labels,(1,10)))
    ax2.imshow(np.tile(gt,(1,10)))
    ax3.imshow(np.tile(rec,(1,10)))
    axs = [ax0,ax1,ax2,ax3]
    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axis('tight')

    return fig,axs

def grid_fft(grid,sd=False):
    m = grid.shape[0]
    W = np.hamming(m) * np.hamming(m).T
    ham_g = W * grid
    G = np.fft.fftshift(np.fft.fft2(ham_g))
    if sd:
        G = spec_density(G)
    return G

def spec_density(Z):
    m = Z.shape[0]
    Z = np.abs(Z)**2 / m**2
    Z = Z / np.max(Z)
    Z = np_log(Z)
    return Z

def lp_filter(Z,r):
    m = Z.shape[0]
    # Y = np.copy(Z)
    # Y[:r,:] = 0
    # Y[-r:,:] = 0
    # Y[:,:r] = 0
    # Y[:,-r:] = 0
    # reduce the dimension
    X = np.zeros( (m - 2*r, m - 2*r) )
    X = Z[r:-r,r:-r]
    return X
    
def grid_plot_3d(grid,colormap=None,angle_view=(30,60)):
    m = grid.shape[0]
    x = np.linspace(0,1,m)
    X, Y = np.meshgrid(x, x)
    Z = grid
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')

    # Plot the surface.
    if colormap is None:
        colormap = cm.coolwarm
    surf = ax.plot_surface(X, Y, Z, cmap=colormap,
                           linewidth=0, antialiased=False)

    ax.set_xticks(np.linspace(0,1,3))
    ax.set_yticks(np.linspace(0,1,3))

    ax.view_init(*angle_view)

    return fig,ax

def subsample_grid(grid,subsample_method,**kwargs):
    if subsample_method == 'bernoulli':
        return subsample_grid_bernoulli(grid,**kwargs)
    else:
        raise ValueError(f"Unknown subsampling method [{subsample_method}]")
    

def subsample_grid_bernoulli(grid,p=0.5,sort=False):
    m = grid.shape[0]
    sm = npr.binomial(m,p)
    samples = npr.choice(m,size=sm,replace=False)
    og_samples = np.copy(samples)
    if sort:
        samples = np.sort(samples)
    X,Y = np.meshgrid(samples,samples)
    subgrid = grid[X,Y].T
    return subgrid,og_samples

def binarize_grid_bernoulli(grid):
    bgrid = npr.binomial(1,grid)
    return bgrid

def reoder_grid_by_rowvec(grid,sidx):
    X,Y = np.meshgrid(sidx,sidx)
    sgrid = grid[X,Y].T
    return sgrid
    


def value_from_index(index_lookup,values):
    a = []
    for j in range(len(values)):
        i = np.where(index_lookup == values[j])[0]
        a.append(i)
    return a
