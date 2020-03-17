"""
parser for input arguments of

mnist_knn

parsers can get long and ugly.
lets save it for another file.

"""
import argparse

def set_cfg_from_args(cfg,args):
    cfg.model_py = args.model_py
    cfg.model_th = args.model_py
    cfg.epochs = args.model_py
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    cfg.lr = args.lr
    cfg.gamma = args.gamma
    cfg.log_interval = args.log_interval

def parse_args():
    """ Parse input args """
    parser = argparse.ArgumentParser(description="generate results from training a neural network on mnist data for labeling smoothing")
    parser.add_argument('--model_py',type=str,
                        help='model python file path')
    parser.add_argument('--model_th',type=str,
                        help='model torch snapshot path')
    parser.add_argument('--epochs',type=int,default=10,metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--batch_size',type=int,default=16,metavar='N',
                        help='batch size for training')
    parser.add_argument('--seed',type=int,default=1,metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr',type=float,default=1.0,metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',type=float,default=0.7,metavar='M',
                        help='learning rate set gamma (default: 0.7)')
    parser.add_argument('--log_interval',type=int,default=10,metavar='N',
                        help='number of batches before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

