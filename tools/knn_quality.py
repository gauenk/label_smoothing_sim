import numpy as np
import numpy.random as npr
import argparse
from easydict import EasyDict as edict

def load_cfg():
    cfg = edict({})

    cfg.base = edict({})
    cfg.base.size = 100
    cfg.base.random = True
    cfg.base.nclasses = 3
    cfg.base.label_noise = 0.25
    cfg.base.bg_grid_noise = 0.25

    cfg.knn = edict({})
    cfg.knn.target = 3
    cfg.knn.k = 10

    cfg.subset = edict({})
    cfg.subset.method = 'bernoulli'
    cfg.subset.sort = False    
    cfg.subset.bg_grid_noise = 0.25

    return cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    # change the name of the saved model
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("Called with args")
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    else:
        cfg = load_cfg()


if __name__ == __main__:
    print("HI")
    main()
