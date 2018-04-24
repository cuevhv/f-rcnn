import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.MODEL = edict()
__C.MODEL.TRAIN = False
__C.MODEL.POOL5 = True
__C.MODEL.VGG16_FC = True
