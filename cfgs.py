# version
VERSION = 'test_2'

# dataset
ALL_MEANS = [0.473, 0.441, 0.424]
ALL_STD = [0.256, 0.257, 0.250]
USED_MEANS = [0.478, 0.447, 0.430]
USED_STD = [0.256, 0.258, 0.250]

# common
CLASS_FILE = './data/classes/hand_name.txt'
NUM_CLASS = 3
INPUT_IMAGE_H = 448
INPUT_IMAGE_W = 448
DOWN_RATIO = 4
MAX_OBJ = 10

# train
TRAIN_DATA_FILE = './data/dataset/hand_train.txt'
BATCH_SIZE = 8
EPOCHS = 80
MAX_KEEP = 10
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
USE_AUG = False
# loss weight
HM_LOSS_WEIGHT = 1
WH_LOSS_WEIGHT = 0.05
REG_LOSS_WEIGHT = 1


# test
TEST_DATA_FILE = './data/dataset/hand_test.txt'
SCORE_THRESHOLD = 0.2
USE_NMS = True
NMS_THRESH = 0.5
SHOW_NUM = 2
