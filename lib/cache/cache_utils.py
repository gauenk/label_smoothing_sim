from configBase import experiment_config_from_cfg_file
from cache.model_cache import ModelCache

def model_cache_from_cfg_file(cfg_file):
    exp_cfg = experiment_config_from_cfg_file(cfg_file)
    model_cache = ModelCache(exp_cfg.MODEL_CACHE_ROOT,exp_cfg,model_index,None,"model_info")
    return model_cache

