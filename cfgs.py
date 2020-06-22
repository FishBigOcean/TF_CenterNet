import os
import tensorflow as tf

# version
VERSION = 'v1.4_1'

# dataset
USED_MEANS = [0.490, 0.451, 0.429]
USED_STD = [0.251, 0.249, 0.242]

# common
CLASS_FILE = './data/classes/hand_name.txt'
NUM_CLASS = 3
INPUT_IMAGE_H = 288  # 448 352 288
INPUT_IMAGE_W = 288  # 448 352 288
DOWN_RATIO = 4
MAX_OBJ = 10

# train
TRAIN_DATA_FILE = './data/dataset/train-v3.txt'
BATCH_SIZE = 16
EPOCHS = 40
MAX_KEEP = 10
SAVE_MIN = True
WEIGHT_INITIALIZER = tf.variance_scaling_initializer(2, 'fan_avg', 'truncated_normal')
WEIGHT_REGULARIZER = tf.contrib.layers.l2_regularizer(0.0005)
BN_MOMENTUM = 0.99
MBV3_SHRINK = 1
# learning rate
LR_TPYE = "piecewise"  # "exponential","piecewise","CosineAnnealing"
LR = 1e-3  # exponential
LR_DECAY_STEPS = 5000  # exponential
LR_DECAY_RATE = 0.95  # exponential
LR_BOUNDARIES = [40000, 50000]  # piecewise
LR_PIECEWISE = [0.001, 0.0003, 0.0001]  # piecewise
WARM_UP_EPOCHS = 2  # CosineAnnealing
INIT_LR = 1e-4  # CosineAnnealing
END_LR = 1e-6  # CosineAnnealing
PRE_TRAIN = False
USE_AUG = True
# loss weight
HM_POS_WEIGHT = 2
HM_NEG_WEIGHT = 1
HM_LOSS_WEIGHT = 1
WH_LOSS_WEIGHT = 0.1
REG_LOSS_WEIGHT = 1
CLS_LOSS_WEIGHT = 1
SIGMA = 10

# test
TEST_DATA_FILE = './data/dataset/test-v3.txt'
SCORE_THRESHOLD = 0.2
USE_NMS = True
NMS_THRESH = 0.5
SHOW_NUM = 1
VAL_IOU_THRESH = 0.6
