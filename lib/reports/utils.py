"""
Helper functions for managing the loading of the saved results
"""

# python imports
import sys,os,re,glob,copy
import os.path as osp
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

# project imports
from base.utils import read_pickle
from base.config import readYamlToEdict

def load_cls_info_files_final(set_str,exp_uuid,root_dir,cls_info_fn):
    experiments = {}
    fdir = osp.join(root_dir,exp_uuid,"results_24")
    fpath = osp.join(fdir,cls_info_fn.format(set_str))
    if osp.exists(fpath):
        experiments[20] = edict(read_pickle(fpath))
    return experiments

def load_cls_info_files_all(set_str,exp_uuid,root_dir,cls_info_fn):
    experiments = {}
    exp_path = osp.join(root_dir,exp_uuid)
    results_glob = osp.join(exp_path,"results_*")
    results_regex = osp.join(exp_path,"results_(?P<n>[0-9]+)")
    for results_i in glob.glob(results_glob):
        m = re.match(results_regex,results_i)
        index = int(m.groupdict()['n'])
        fpath = osp.join(results_i,cls_info_fn.format(set_str))
        experiments[index] = edict(read_pickle(fpath))
    return experiments

def load_cls_info_files(select,set_str,*args):
    if select == "final" or select == "f":
        return load_cls_info_files_final(set_str,*args)
    elif select == "all" or select == "a":
        return load_cls_info_files_all(set_str,*args)        
    else:
        raise KeyError("[reports/utils.py] No selection option{}".format(select))

def load_cls_infos(exp_uuids,root_dir,cls_info_fn):
    N = len(exp_uuids)
    cls_infos_tr = dict.fromkeys(exp_uuids)
    cls_infos_te = dict.fromkeys(exp_uuids)
    for exp_uuid in exp_uuids:
        file_info = [exp_uuid,root_dir,cls_info_fn]
        cls_infos_tr[exp_uuid] = load_cls_info_files('f','tr',*file_info)
        cls_infos_te[exp_uuid] = load_cls_info_files('f','te',*file_info)
    cls_infos_tr = remove_empty_elements(cls_infos_tr)
    cls_infos_te = remove_empty_elements(cls_infos_te)
    return cls_infos_tr,cls_infos_te

def remove_empty_elements(dict_obj):
    dict_obj_c = copy.deepcopy(dict_obj)
    for key,item in dict_obj.items():
        if len(dict_obj[key]) == 0:
            del dict_obj_c[key]
    for key,item in dict_obj_c.items():
        print(key,item[20])
    return dict_obj_c

def load_configs(exp_uuids,root_dir):
    N = len(exp_uuids)
    configs = dict.fromkeys(exp_uuids)
    for exp_uuid in exp_uuids:
        fpath = osp.join(root_dir,exp_uuid,'config.yml')
        configs[exp_uuid] = readYamlToEdict(fpath)
    return configs

