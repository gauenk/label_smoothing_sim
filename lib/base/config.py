import yaml,os
import numpy as np
import os.path as osp
from easydict import EasyDict as edict

def save_cfg(cfg):
    pdir = get_cfg_checkpoint_dir(cfg)
    if not osp.exists(pdir):
        os.makedirs(pdir)
    path = osp.join(pdir, "config.yml")
    print("WRITING CFG: ",path)
    wstr = yaml.dump(cfg,default_flow_style=False)
    with open(path,'w') as f:
        f.write(wstr)

def write_cfg_file(cfg,filepath):
    wstr = yaml.dump(cfg,default_flow_style=False)
    with open(filepath,'w') as f:
        f.write(wstr)

def merge_a_into_b(a,b):
    return _merge_a_into_b(a, b)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b.keys():
            print(b.keys())
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(b[k])
        # the types must match, too; unless old_type is not edict and not None; and new_type is not None
        if old_type is not type(v) and \
        (old_type is edict and old_type is not type(None))\
        and type(v) is not type(None):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        elif v == "None":
            b[k] = None
        else:
            b[k] = v

# misc functions... not sure where they should live

def experiment_config_from_cfg_file(cfg_file):
    exp_config = readYamlToEdict(cfg_file)
    # merged_config = magic_merge_here(....)
    merged_config = None
    return merged_config

def read_cfg_file(cfg_file):
    return readYamlToEdict(cfg_file)

def readYamlToEdict(yaml_file):
    import yaml
    with open(yaml_file, 'r') as f:
        yaml_cfg = edict(yaml.load(f,Loader=yaml.FullLoader))
    return yaml_cfg

def checkListEqualityWithOrder(list_a,list_b):
    print(list_a)
    if len(list_a) == 0 and len(list_b) == 0:
        return True
    if len(list_a) != len(list_b):
        return False
    for item_a in list_a:
        for item_b in list_b:        
            if type(item_a) is list and type(item_b) is list:
                if not checkListEqualityWithOrder(item_a,item_b):
                    return False
            elif type(item_a) is edict and type(item_b) is edict:
                if checkEdictEquality(item_a,item_b):
                    return False
            elif type(item_a) is dict and type(item_b) is dict:
                if checkEdictEquality(item_a,item_b):
                    return False
            elif item_a != item_b:
                return False
    return True


def all_true_in_list(boolList):
    for boolValue in boolList:
        if boolValue is False:
            return False
    return True
    
def any_true_in_list(boolList):
    for boolValue in boolList:
        if boolValue is True:
            return True
    return False
            
def checkNdarrayEqualityWithOrder(ndarray_a,ndarray_b):
    return np.all(ndarray_a == ndarray_b)

def checkListEqualityWithOrderIgnored(list_a,list_b):
    # all the elements in list_a are somewhere in list_b
    if len(list_a) == 0 and len(list_b) == 0:
        return True
    if len(list_a) != len(list_b):
        return False
    for item_a in list_a:
        boolList = []
        for item_b in list_b:        
            if type(item_a) is list and type(item_b) is list:
                if checkListEqualityWithOrderIgnored(item_a,item_b):
                    boolList.append(True)
                else:
                    boolList.append(False)
            elif type(item_a) is edict and type(item_b) is edict:
                if checkEdictEquality(item_a,item_b):
                    boolList.append(True)
                else:
                    boolList.append(False)
            elif type(item_a) is dict and type(item_b) is dict:
                if checkEdictEquality(item_a,item_b):
                    boolList.append(True)
                else:
                    boolList.append(False)
            elif item_a != item_b:
                boolList.append(False)
            else:
                boolList.append(True)
        if not any_true_in_list(boolList):
            return False
    return True

def check_list_equality(alist,blist):
    for a,b in zip(alist,blist):
        if type(a) != type(b):
            return False
        if type(a) is list:
            isValid = check_list_equality(a,b)
            if not isValid:
                return False
    return True

def checkEdictEquality(validConfig,proposedConfig):
    return checkConfigEquality(validConfig,proposedConfig)
    # """
    # check if the input config edict is the same
    # as the current config edict
    # """
    # for key,validValue in validConfig.items(): # iterate through the "truth"
    #     if key not in proposedConfig.keys():
    #         return False
    #     proposedValue = proposedConfig[key]
    #     if type(validValue) is list:
    #         isValid = checkListEqualityWithOrderIgnored(validValue,proposedValue)
    #         if isValid is False:
    #             return False
    #         continue
    #     if type(validValue) is edict or type(validValue) is dict:
    #         isValid = checkEdictEquality(validValue,proposedValue)
    #         if isValid is False:
    #             return False
    #         continue
    #     if type(validValue) is np.ndarray:
    #         if type(proposedValue) is np.ndarray:
    #             isValid = checkNdarrayEqualityWithOrder(validValue,proposedValue)
    #             if isValid is False:
    #                 return False
    #             continue
    #         else:
    #             return False
    #     if proposedValue != validValue:
    #         return False
    # return True

def checkConfigEquality(validConfig,proposedConfig):
    """
    check if the input config edict is the same
    as the current config edict
    """
    for key,validValue in validConfig.items(): # iterate through the "truth"
        if key in ['_DEBUG','BIJECTION']:
            continue

        if key not in proposedConfig.keys():
            # print("not found")
            # print(key)
            # print(validValue)
            return False
        proposedValue = proposedConfig[key]
        if type(validValue) is edict or type(validValue) is dict:
            #print(key)
            isValid = checkConfigEquality(validValue,proposedValue)
            if not isValid:
                # print("recurse")
                # print(key)
                # print(validValue)
                # print(proposedValue)
                return False
            continue

        if type(proposedValue) in [list,np.ndarray]:
            if type(validValue) not in [list,np.ndarray]:
                return False
            isValid = check_list_equality(proposedValue,validValue)
            if not isValid:
                return False
            continue

        if proposedValue != validValue:
            # print(key)
            # print(type(validValue),type(proposedConfig[key]))
            # print(validValue)
            # print(proposedConfig[key])
            return False
    return True

def split_cfg_to_aux_and_global(cfg_file):
    _cfg = readYamlToEdict(cfg_file)
    aux_cfg = {}
    global_cfg = {}
    for key,value in _cfg.items():
        if key.startswith("_"):
            new_key = key[1:]
            aux_cfg[new_key] = value
        else:
            global_cfg[key] = value
    return edict(aux_cfg),edict(global_cfg)

def cfg_from_file(filename,merge_to=None):
    """Load a config file and merge it into the default options."""
    aux_cfg,yaml_cfg = split_cfg_to_aux_and_global(filename)
    cfg_from_edict(yaml_cfg,merge_to=merge_to)
    return aux_cfg

def cfg_from_edict(edict_obj,merge_to=None):
    """Load a config file and merge it into the default options."""
    if merge_to is None:
        merge_a_into_b(edict_obj, __C)
    else:
        merge_a_into_b(edict_obj, merge_to) 
    load_tp_fn_record_path()

def get_cfg_checkpoint_dir(cfg):
    return f"output/{cfg.output_dir}/{cfg.exp_name}/"

def get_cfg_checkpoint_fn(cfg):
    return osp.join(get_cfg_checkpoint_dir(cfg),"model_params_{}.th")

