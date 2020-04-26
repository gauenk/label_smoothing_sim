"""
This is code to apply the graph kernel from 
"learning with local and global consistency", Zhou et al 2004
"""


# python imports
import sys,os
import numpy as np
import numpy.linalg as npl
from sklearn.metrics import pairwise_distances

# project imports

def np_matdiff(A,B):
    return np.sum(np.abs(A-B))

def buildAdjacencyMatrix(features,sigma2=2):
    diff = pairwise_distances(features,metric="l2")
    W = np.exp( - diff / (2 * sigma2) ) - np.eye( diff.shape[0] )
    return W

def buildLaplacian(W):
    D = np.diag(np.sum(W,axis=1))
    Dsqrt = np.sqrt(D)
    Dsqrt_inv = npl.inv(Dsqrt)
    S = Dsqrt_inv @ W @ Dsqrt_inv
    return S

def approximateInv(S,Y,alpha,tol=10**-6):
    F = np.copy(Y)
    Fp = 10 * np.copy(F)
    while( np_matdiff(Fp,F) > tol ):
        Fp = np.copy(F)
        F = alpha * np.matmul(S, F) + (1 - alpha) * Y
    return F

def graphKernel2006(cfg,outputs,features):
    # match varaible names with algorithm from pdf
    W = buildAdjacencyMatrix(features)
    S = buildLaplacian(W) 
    Y = outputs
    F_star = approximateInv(S,Y,cfg.alpha)
    guesses = np.argmax(outputs,axis=1).astype(np.int)
    guesses_new = np.argmax(F_star,axis=1).astype(np.int)
    return np.argmax(F_star,axis=1)
