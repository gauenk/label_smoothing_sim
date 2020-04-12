import os,uuid,copy
import os.path as osp
from easydict import EasyDict as edict

from utils.base import readPickle,writePickle
from cache.two_level_cache import TwoLevelCache

"""
roidbCacheCfg = edict() # set of parameters to follow when loading the dataset from the lookup cache
roidbCacheCfg.CACHE_PROMPT = edict()
roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION_RANDOMIZE = True
roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION_RANDOMIZE_SUBSET = True
roidbCacheCfg.CACHE_PROMPT.DATASET.SUBSAMPLE_BOOL = True
"""

class ModelCache(TwoLevelCache):

    def __init__(self,loaded_config):
        self.model_cache_config = loaded_config
        self.root_dir = loaded_config.MODEL_CACHE_DIR
        if self.root_dir in [None,False,""]:
            print(__file__,"root_dir from MODEL_CACHE_DIR is invalid")
            exit()
        super(ModelCache,self).__init__(self.root_dir,loaded_config,None,"model_states","states")

    # def construct_model_cache_config(self,exp_cfg,model_index):
    #     model_cache_config = edict()

    #     model_cache_config.id = "default_id"
    #     model_cache_config.task = exp_cfg.TASK
    #     model_cache_config.subtask = exp_cfg.SUBTASK
    #     model_cache_config.model_info = exp_cfg.MODEL_INFO_LIST[model_index]
    #     # we don't want to save the actual config within the modelInfo
    #     model_cache_config.model_info.additional_input.info['activations'].exp_cfg = None

    #     # set training arguments by the task
    #     if model_cache_config.task == 'detection':
    #         model_cache_config.train_args = exp_cfg.TRAIN.OBJ_DET
    #     elif model_cache_config.task == 'classification':
    #         model_cache_config.train_args = exp_cfg.TRAIN.CLS
    #     elif model_cache_config.task == 'regression':
    #         model_cache_config.train_args = exp_cfg.TRAIN.REGRESSION
    #     elif model_cache_config.task == 'generation':
    #         model_cache_config.train_args = exp_cfg.TRAIN.GENERATION

    #     return model_cache_config

    # def print_dataset_summary_by_uuid(datasetConfigList,uuid_str):
    #     pass
