"""
This model runs and aggregates results from
several experiments.

"""

# python imports
import sys,os,copy,glob,uuid
import os.path as osp
import multiprocessing
from multiprocessing import Pool
from easydict import EasyDict as edict

# project imports
import _init_paths
import settings
from nn_knn.utils import load_results
from base.config import write_cfg_file,read_cfg_file
from datasets.mnist import load_mnist_cfg
from datasets.cifar import load_cifar_cfg
from reports import acc_vs_noise
import mnist_knn,cifar_knn
import mnist_knn_parser

mnist_exp_set = [
    "3b28c495-bc23-4b84-af7c-d8f74559ebff",
    "e83ee5fe-c5e7-456c-9c12-f1b59d803525",
    "f1f8c86d-3f83-4e50-a584-f495522e4411",
    "df4dc3a7-8c4b-4897-8212-72cd3b38d64f",
    "2b385b4e-bffc-44a2-9fa8-1f916262925b",
    "74ed0198-e1a8-41ba-87d5-814672f525f1",
]

cifar_exp_set = [
    "e3a00f51-41e0-40a4-840a-36018610d6a1",
    "0902a486-d9cc-4fd4-a5eb-2bb1a5f8cc34",
    "0487e813-dd4f-49eb-a2d3-0178b472da7c",
    
]



def get_cfg_dir():
    return osp.join(settings.ROOT_PATH,'output/run_exp_set/')

def get_uuid_cache_path():
    exp_dir = osp.join(settings.ROOT_PATH,'output/run_exp_set')
    if not osp.exists(exp_dir):
        os.makedirs(exp_dir)
    exp_names_cache = osp.join(exp_dir,"uuid_cache.txt")
    return exp_names_cache

def load_exp_uuids_cache_from_yaml():
    exp_uuids = []
    path = get_cfg_dir()
    for exp_uuid_f in glob.glob(osp.join(path,"./tmp_runme_*")):
        config = read_cfg_file(exp_uuid_f)
        exp_uuids.append(str(config.exp_name))
    return exp_uuids
    
def load_exp_uuids_glob(path):
    exp_uuids = []
    for exp_uuid_fp in glob.glob(osp.join(path,"*")):
        exp_uuid = str(exp_uuid_fp.split('/')[0-1])
        exp_uuids.append(exp_uuid)
    return exp_uuids

def create_alpha_cfg_list(cfg):
    # alpha_list = [0.,0.5]
    alpha_list = [0.,0.25,0.5,0.75,1.]
    configs = []
    for alpha in alpha_list:
        config = copy.deepcopy(cfg)
        config.graphKernel2006.alpha = alpha
        configs.append(config)
    return configs

def create_noise_cfg_list(cfg):
    # noise_list = [0.25,0.5]
    noise_list = [0.,0.10,0.25,0.35,0.5,0.65,0.75,.90,.95,.99]
    configs = []
    for noise in noise_list:
        config = copy.deepcopy(cfg)
        config.label_noise = noise
        configs.append(config)
    return configs

def generate_config_set(cfg,stratifyFunctions):
    if len(stratifyFunctions) == 0:
        return [cfg]
    function = stratifyFunctions.pop()
    config_fset = function(cfg)
    # check_config_set(config_fset)
    config_gset_list = []
    for config in config_fset:
        stratifyFunctions_c = copy.copy(stratifyFunctions)
        config_gset = generate_config_set(config,stratifyFunctions_c)
        config_gset_list.extend(config_gset)
    return config_gset_list

def check_config_set(cfg_list):
    print(len(cfg_list))
    print(len(cfg_list[0]))
    for cfg in cfg_list:
        print(cfg.graphKernel2006.alpha,cfg.label_noise)

def run_cmd(index):
    args = mnist_knn_parser.parse_args(force=True)
    cdir = get_cfg_dir()
    cfg_file = osp.join(cdir,'tmp_runme_{}.yml'.format(index))
    args.cfg_file = cfg_file
    #mnist_knn.main(args)
    cifar_knn.main(args)

def set_exp_name(config):
    config.exp_name = uuid.uuid4()

def execute_config_set(config_set):
    """
    Run the set of experiments
    """

    qsize = 3
    curr_q = []
    # cmd = "cifar_knn.py --cfg_file {}"
    exp_names = []
    for idx,config in enumerate(config_set):
        #
        # -- setup the config with name --
        #
        cdir = get_cfg_dir()
        cfg_fn_i = osp.join(cdir,"tmp_runme_{}.yml".format(idx))
        set_exp_name(config)
        exp_names.append(str(config.exp_name))
        write_cfg_file(config,cfg_fn_i)
        print(config.exp_name)

        #
        # -- run the multiprocess --
        #
        p = multiprocessing.Process(target=run_cmd,args=(idx,))
        curr_q.append(p)
        if len(curr_q) == qsize:
            for p in curr_q:
                p.start()
            for p in curr_q:
                p.join()
            curr_q = []
            write_exp_uuids_cache(exp_names)

    if len(curr_q) > 0:
            for p in curr_q:
                p.start()
            for p in curr_q:
                p.join()
            write_exp_uuids_cache(exp_names)

    # save the exp names
    #print(exp_names)
    write_exp_uuids_cache(exp_names)
    return exp_names

def clear_exp_uuids_cache():
    print("clear exp uuids cache")
    exp_names_cache = get_uuid_cache_path()
    with open(exp_names_cache,'w') as f:
        f.write()
    
def update_exp_uuids_cache(exp_names):
    print("append exp uuids cache: not used in code lol")
    exp_names_cache = get_uuid_cache_path()
    with open(exp_names_cache,'a') as f:
        f.write('\n'.join(exp_names) + '\n')

def write_exp_uuids_cache(exp_names):
    print("writing exp uuids cache")
    exp_names_cache = get_uuid_cache_path()
    with open(exp_names_cache,'w') as f:
        f.write('\n'.join(exp_names) + '\n')

def load_exp_uuids_cache():
    exp_dir = osp.join(settings.ROOT_PATH,'output/run_exp_set')
    exp_names_cache = osp.join(exp_dir,"uuid_cache.txt")
    with open(exp_names_cache,'r') as f:
        exp_names = f.readlines()
    exp_names = [uuid.strip() for uuid in exp_names]
    print("loaded uuid cache")
    print(exp_names)
    return exp_names

def main():

    root_dir = osp.join(settings.ROOT_PATH,"output/cifar_knn/")
    #
    # -- load base config --
    #

    cfg = load_cifar_cfg()
    # cfg = load_mnist_cfg()

    #
    # -- create experiment set --
    #

    stratifyFunctions = [create_noise_cfg_list,create_alpha_cfg_list]
    config_set = generate_config_set(cfg,stratifyFunctions)
    # check_config_set(config_set)

    #
    # -- execute code --
    #

    exp_uuids = execute_config_set(config_set)
    # exp_uuids = load_exp_uuids_glob(root_dir)
    # exp_uuids = load_exp_uuids_cache_from_yaml()

    #
    # -- run results on exp_uuid set --
    #


    print("Plotting results")
    exp_list = []
    # exp_uuids = ["74d68373-48d1-4ada-89e3-353305c83ee0"]

    # root_dir = osp.join(settings.ROOT_PATH,"output/mnist_knn/")
    # exp_uuids = load_exp_uuids_cache(root_dir)
    # exp_uuids = load_exp_uuids(root_dir)
    # exp_uuids = mnist_exp_set
    cls_info_fn = "cls_info_{}.pkl"
    acc_vs_noise.main( exp_uuids,root_dir,cls_info_fn)

    # # load in results
    
    
    # # generate plots
    
        
if __name__ == "__main__":
    main()

