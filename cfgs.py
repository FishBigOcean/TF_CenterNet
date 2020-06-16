import os
import tensorflow as tf

# version
VERSION = 'test_4_ori'

# dataset
ALL_MEANS = [0.471, 0.443, 0.425]
ALL_STD = [0.251, 0.254, 0.247]
USED_MEANS = [0.471, 0.443,  0.425]
USED_STD = [0.250, 0.253, 0.245]

# common
CLASS_FILE = './data/classes/hand_name.txt'
NUM_CLASS = 3
INPUT_IMAGE_H = 352  # 448 384  352
INPUT_IMAGE_W = 352  # 448 384  352
DOWN_RATIO = 4
MAX_OBJ = 10

# train
TRAIN_DATA_FILE = './data/dataset/hand_train_new.txt'
BATCH_SIZE = 16
EPOCHS = 70
MAX_KEEP = 10
SAVE_MIN = True
WEIGHT_REGULARIZER = tf.contrib.layers.l2_regularizer(0.0005)
BN_MOMENTUM = 0.99
# learning rate
LR_TPYE = "piecewise"  # "exponential","piecewise","CosineAnnealing"
LR = 1e-3  # exponential
LR_DECAY_STEPS = 5000  # exponential
LR_DECAY_RATE = 0.95  # exponential
LR_BOUNDARIES = [40000, 60000]  # piecewise
LR_PIECEWISE = [0.0001, 0.00001, 0.000001]  # piecewise
WARM_UP_EPOCHS = 2  # CosineAnnealing
INIT_LR = 1e-4  # CosineAnnealing
END_LR = 1e-6  # CosineAnnealing
PRE_TRAIN = False
USE_AUG = True
# loss weight
HM_LOSS_WEIGHT = 1
WH_LOSS_WEIGHT = 0.1
REG_LOSS_WEIGHT = 1
SIGMA = 1.0

# test
TEST_DATA_FILE = './data/dataset/hand_test_new.txt'
SCORE_THRESHOLD = 0.2
USE_NMS = True
NMS_THRESH = 0.5
SHOW_NUM = 1
