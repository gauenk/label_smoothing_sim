"""
parser for input arguments of

mnist_knn

parsers can get long and ugly.
lets save it for another file.

"""
import argparse,sys

def set_cfg_from_args(cfg,args):
    cfg.model_py = set_if_not_none(cfg,args,'model_py')
    cfg.model_th = set_if_not_none(cfg,args,'model_th')
    cfg.epochs = set_if_not_none(cfg,args,'epochs')
    cfg.batch_size = set_if_not_none(cfg,args,'batch_size')
    cfg.seed = set_if_not_none(cfg,args,'seed')
    cfg.lr = set_if_not_none(cfg,args,'lr')
    cfg.gamma = set_if_not_none(cfg,args,'gamma')
    cfg.log_interval = set_if_not_none(cfg,args,'log_interval')

def set_if_not_none(cfg,args,fieldname):
    if args.__dict__[fieldname] is not None:
        return args.__dict__[fieldname]
    else:
        return cfg[fieldname]

def parse_args():
    """ Parse input args """
    parser = argparse.ArgumentParser(description="generate results from training a neural network on mnist data for labeling smoothing")
    parser.add_argument('--model_py',type=str,default=None,
                        help='model python file path')
    parser.add_argument('--model_th',type=str,default=None,
                        help='model torch snapshot path')
    parser.add_argument('--epochs',type=int,default=None,metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--batch_size',type=int,default=None,metavar='N',
                        help='batch size for training')
    parser.add_argument('--seed',type=int,default=None,metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr',type=float,default=None,metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',type=float,default=None,metavar='M',
                        help='learning rate set gamma (default: 0.7)')
    parser.add_argument('--log_interval',type=int,default=None,metavar='N',
                        help='number of batches before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--default', action='store_true', default=False,
                        help='Just run me on default :D')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

