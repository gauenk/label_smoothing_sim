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

class TrainResultsCache(TwoLevelCache):

    def __init__(self,root_dir,cfg,imdb_config,roidb_settings,lookup_id):
        self.train_cache_config = self.construct_data_cache_config(cfg,imdb_config)
        super(TrainResultsCache,self).__init__(root_dir,self.train_cache_config,roidb_settings,lookup_id,"train_results")

    def reset_dataset_augmentation(self,datasetAugmentationCfg):
        temporaryConfig = copy.deepcopy(self.trainCacheConfig)
        self.set_dataset_augmentation(temporaryConfig,datasetAugmentationCfg)
        self.update_config(temporaryConfig)

    def set_dataset_augmentation(self,train_cache_config,datasetAugmentationCfg):
        train_cache_config.dataset_augmentation = edict() # primary config set 1
        train_cache_config.dataset_augmentation.bool_value = datasetAugmentationCfg.BOOL
        #train_cache_config.dataset_augmentation.configs = datasetAugmentationCfg.CONFIGS # says: "what type of augmentations do we have?"
        train_cache_config.dataset_augmentation.image_translate = datasetAugmentationCfg.IMAGE_TRANSLATE
        train_cache_config.dataset_augmentation.image_rotate = datasetAugmentationCfg.IMAGE_ROTATE
        train_cache_config.dataset_augmentation.image_crop = datasetAugmentationCfg.IMAGE_CROP
        train_cache_config.dataset_augmentation.image_flip = datasetAugmentationCfg.IMAGE_FLIP
        train_cache_config.dataset_augmentation.percent_augmentations_used = datasetAugmentationCfg.N_PERC # says: "how many of the possible augmentations should we use for each augmented sample?"
        train_cache_config.dataset_augmentation.percent_samples_augmented = datasetAugmentationCfg.N_SAMPLES # says: "how many samples are we augmenting?"
        # train_cache_config.dataset_augmentation.bool_by_samples = datasetAugmentationCfg.SAMPLE_BOOL_VECTOR # says: which samples are augmented?
        train_cache_config.dataset_augmentation.randomize = datasetAugmentationCfg.RANDOMIZE
        train_cache_config.dataset_augmentation.randomize_subset = datasetAugmentationCfg.RANDOMIZE_SUBSET
        

    def construct_data_cache_config(self,cfg,imdb_config):
        trainCacheConfig = edict()

        trainCacheConfig.id = "default_id"
        trainCacheConfig.task = cfg.TASK
        trainCacheConfig.subtask = cfg.SUBTASK
        trainCacheConfig.modelInfo = cfg.modelInfo
        trainCacheConfig.transform_each_sample = cfg.DATASETS.TRANSFORM_EACH_SAMPLE
        # we don't want to save the actual config within the modelInfo
        trainCacheConfig.modelInfo.additional_input.info['activations'].cfg = None

        self.set_dataset_augmentation(trainCacheConfig,cfg.DATASET_AUGMENTATION)
        trainCacheConfig.dataset = edict() # primary config set 2
        trainCacheConfig.dataset.name = cfg.DATASETS.CALLING_DATASET_NAME
        trainCacheConfig.dataset.imageset = cfg.DATASETS.CALLING_IMAGESET_NAME
        trainCacheConfig.dataset.config = cfg.DATASETS.CALLING_CONFIG
        trainCacheConfig.dataset.subsample_bool = cfg.DATASETS.SUBSAMPLE_BOOL
        trainCacheConfig.dataset.annotation_class = cfg.DATASETS.ANNOTATION_CLASS
        # trainCacheConfig.dataset.size = cfg.DATASETS.SIZE #len(roidb) ## We can't use this because it is set by the unfiltered image_index variable
        # trainCacheConfig.dataset.classes = cfg.DATASETS.CLASSES #imdb.classes; classes should already be filtered if they need to be ## can't use this it is unfiltered

        
        trainCacheConfig.filters = edict() # primary config set 3
        trainCacheConfig.filters.classes = cfg.DATASETS.FILTERS.CLASS
        trainCacheConfig.filters.empty_annotations = cfg.DATASETS.FILTERS.EMPTY_ANNOTATIONS

        trainCacheConfig.misc = edict() # primary config set 3
        trainCacheConfig.misc.use_diff = imdb_config['use_diff']
        trainCacheConfig.misc.rpn_file = imdb_config['rpn_file']
        trainCacheConfig.misc.min_size = imdb_config['min_size']
        trainCacheConfig.misc.flatten_image_index = imdb_config['flatten_image_index']
        trainCacheConfig.misc.setID = imdb_config['setID']

        return trainCacheConfig
        # TODO: put theses somewhere else
        # ? assert len(roidb) == trainCacheConfig.DATASET.SIZE
        # ? assert imdb.classes == trainCacheConfig.DATASET.CLASSES

    def print_dataset_summary_by_uuid(datasetConfigList,uuid_str):
        pass
