from easydict import EasyDict as edict


def get_cfg_graphKernel_2006(alpha=0.5):
    cfg_graphKernel_2006 = edict()
    cfg_graphKernel_2006.alpha = alpha
    cfg_graphKernel_2006.sigma2 = 1.
    return cfg_graphKernel_2006
