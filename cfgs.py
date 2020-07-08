import os
import tensorflow as tf

# version
VERSION = 'd_1'

# dataset
USED_MEANS = [0.490, 0.451, 0.429]
USED_MEANS_255 = [125, 115, 109]
USED_STD = [0.251, 0.249, 0.242]

# common
CLASS_FILE = './data/classes/hand_name.txt'
NUM_CLASS = 1
INPUT_IMAGE_H = 256  # 448 352 288
INPUT_IMAGE_W = 256  # 448 352 288
DOWN_RATIO = 4
MAX_OBJ = 3
ADD_REG = True

# train
TRAIN_DATA_FILE = './data/dataset/train-v4.txt'
BATCH_SIZE = 32
TRAIN_BATCH = 336
EPOCHS = 400
MAX_KEEP = 20
SAVE_MIN = True
WEIGHT_INITIALIZER = tf.variance_scaling_initializer(2, 'fan_avg', 'truncated_normal')
WEIGHT_REGULARIZER = tf.contrib.layers.l2_regularizer(0.0005)
BN_MOMENTUM = 0.99
MBV3_SHRINK = 1
CSPNET_SHRINK = 0.5
# learning rate
LR_TPYE = "cosine_decay_restarts"  # "exponential","piecewise","CosineAnnealing","cosine_decay_restarts"
# exponential
LR = 1e-3
LR_DECAY_STEPS = 5000
LR_DECAY_RATE = 0.95
# piecewise
LR_BOUNDARIES = [80 * TRAIN_BATCH, 120 * TRAIN_BATCH]
LR_PIECEWISE = [0.003, 0.0003, 0.00003]
# CosineAnnealing
WARM_UP_EPOCHS = 2
INIT_LR = 1e-3
END_LR = 1e-6
# cosine_decay_restarts
WARM_UP_EPOCHS_CR = 10
INIT_LR_CR = 3e-3
END_LR_CR = 1e-6
FIRST = 110 * TRAIN_BATCH
T_MUL = 1.2
M_MUL = 1.

PRE_TRAIN = False
USE_AUG = True
USE_ROTATE = True
# loss weight
HM_POS_WEIGHT = 1
HM_NEG_WEIGHT = 1
HM_LOSS_WEIGHT = 1
WH_LOSS_WEIGHT = 1
REG_LOSS_WEIGHT = 1
SIGMA = 10

# test
TEST_DATA_FILE = './data/dataset/test-v6.txt'
SCORE_THRESHOLD = 0.4
USE_NMS = False
NMS_THRESH = 0.5
SHOW_NUM = 1
VAL_IOU_THRESH = 0.6
