# --------------------------------------------------------
# Configurations for domain adaptation
#
# Written by Shiyu Tang
# Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py
# --------------------------------------------------------

import os.path as osp

import numpy as np
from easydict import EasyDict

from advent.utils import project_root
from advent.utils.serialization import yaml_load


cfg = EasyDict()

# COMMON CONFIGS
cfg.name = '0719_addcontra_clossw0.001_0.07temp_0.9momentum_moco'
# source domain
cfg.SOURCE = 'GTA5'
# target domaino'p
cfg.TARGET = 'Cityscapes'
# Number of workers for dataloading
cfg.NUM_WORKERS = 4
# List of training images
cfg.DATA_LIST_SOURCE = str(project_root / 'advent/dataset/gta5_list/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'advent/dataset/cityscapes_list/{}.txt')
cfg.DATA_LIST_SOURCE_SYNTHIA = str(project_root / 'advent/dataset/synthia_list/{}.txt')
# Directories
cfg.DATA_DIRECTORY_SOURCE = str(project_root / '../data/GTA5')
cfg.DATA_DIRECTORY_TARGET = str(project_root / '../data/cityscapes')
cfg.DATA_DIRECTORY_SOURCE_SYNTHIA = str(project_root / '../data/Synthia')
cfg.DATA_DIRECTORY_STYLE = str(project_root / '../data/cityscapes/train/ambulance')
# Number of object classes
cfg.NUM_CLASSES = 19
# Exp dirs
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / 'experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.GPU_ID = "cuda:0"

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'all'
cfg.TRAIN.SET_SOURCE_SYNTHIA = 'train'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.BATCH_SIZE_STYLE = 4
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 640)
cfg.TRAIN.INPUT_SIZE_TARGET = (1280, 640)
cfg.TRAIN.INPUT_SIZE_STYLE = (1280, 640)
cfg.TRAIN.INPUT_SIZE_TARGET = (1280, 640)
# Class info
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.INFO_TARGET = str(project_root / 'advent/dataset/cityscapes_list/info.json')
# Segmentation network params
cfg.TRAIN.MODEL = 'DeepLabv2'
cfg.TRAIN.MULTI_LEVEL = True
cfg.TRAIN.RESTORE_FROM = ''
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.IMG_MEAN_style = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.uint8)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1  # weight of conv4 prediction. Used in multi-level setting.
# Domain adaptation
cfg.TRAIN.DA_METHOD = 'AdvEnt'
# Adversarial training params
cfg.TRAIN.baseline = False
cfg.TRAIN.LEARNING_RATE_D = 1e-4
cfg.TRAIN.LAMBDA_ADV_MAIN = 0.001
cfg.TRAIN.LAMBDA_ADV_AUX = 0.0002
# MinEnt params
cfg.TRAIN.LAMBDA_ENT_MAIN = 0.001
cfg.TRAIN.LAMBDA_ENT_AUX = 0.0002
# contrastive parameters
cfg.TRAIN.switchcontra = True
cfg.TRAIN.head_mode = 'moco'
cfg.TRAIN.contra_temp = 0.07
cfg.TRAIN.contra_momentum = 0.9
cfg.TRAIN.LAMBDA_CONTRA_S = 0.001
cfg.TRAIN.LAMBDA_CONTRA_T = 0.001
cfg.TRAIN.LAMBDA_CONTRA_T2S = 0.001
cfg.TRAIN.LAMBDA_CONTRA_S2T = 0.001
# cluster parameters
cfg.TRAIN.pseudolabel_cluster = True
cfg.TRAIN.normEuclid = False
cfg.TRAIN.adjthresholdpoly = False
cfg.TRAIN.threshPOWER = 4
cfg.TRAIN.cluster_threshold = 0.05
cfg.TRAIN.ignore_instances = True
# adain parameters
cfg.TRAIN.switchAdain = False
cfg.TRAIN.RESTORE_FROM_decoder = '../../pretrained_models/decoder.pth'
cfg.TRAIN.alpha = 1
cfg.TRAIN.interpolation_weights = [1/4, 1/4, 1/4, 1/4]
# Other params
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 120000
cfg.TRAIN.SAVE_PRED_EVERY = 2000
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 2000
cfg.TRAIN.print_lossrate = 100

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'best'  # {'single', 'best'}
# model
cfg.TEST.MODEL = ('DeepLabv2',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 2000  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 120000  # used in 'best' mode
cfg.TEST.validate_source = False
# Test sets
cfg.TEST.SET_TARGET = 'val'
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / 'advent/dataset/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
