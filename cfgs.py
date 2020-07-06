import os
import tensorflow as tf

# version
VERSION = 'v1.6_4'

# dataset
USED_MEANS = [0.490, 0.451, 0.429]
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
TRAIN_DATA_FILE = './data/dataset/train-v5.txt'
BATCH_SIZE = 16
EPOCHS = 300
MAX_KEEP = 20
SAVE_MIN = True
WEIGHT_INITIALIZER = tf.variance_scaling_initializer(2, 'fan_avg', 'truncated_normal')
WEIGHT_REGULARIZER = tf.contrib.layers.l2_regularizer(0.0005)
BN_MOMENTUM = 0.99
MBV3_SHRINK = 1
CSPNET_SHRINK = 0.5
# learning rate
LR_TPYE = "piecewise"  # "exponential","piecewise","CosineAnnealing"
LR = 1e-3  # exponential
LR_DECAY_STEPS = 5000  # exponential
LR_DECAY_RATE = 0.95  # exponential
LR_BOUNDARIES = [40000, 60000]  # piecewise
LR_PIECEWISE = [0.003, 0.0003, 0.00003]  # piecewise
WARM_UP_EPOCHS = 2  # CosineAnnealing
INIT_LR = 3e-3  # CosineAnnealing
END_LR = 1e-5  # CosineAnnealing
PRE_TRAIN = False
USE_AUG = True
USE_ROTATE = True
# loss weight
HM_POS_WEIGHT = 2
HM_NEG_WEIGHT = 1
HM_LOSS_WEIGHT = 2
WH_LOSS_WEIGHT = 4
REG_LOSS_WEIGHT = 1
SIGMA = 10

# test
TEST_DATA_FILE = './data/dataset/test-v5.txt'
SCORE_THRESHOLD = 0.5
USE_NMS = False
NMS_THRESH = 0.5
SHOW_NUM = 1
VAL_IOU_THRESH = 0.6
