import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import time
import cfgs
from CenterNet import CenterNet
from utils.decode import decode
from utils.image import get_affine_transform, affine_transform
from utils.utils import image_preprocess, py_nms, post_process, bboxes_draw_on_img, read_class_names, cal_iou

ckpt_path = './checkpoint/' + cfgs.VERSION
mode = 3  # 1 GPU 2 CPU 3 single-CPU

if mode == 1:
    sess = tf.Session()
elif mode == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess = tf.Session()
elif mode == 3:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    cpu_num = 1
    config = tf.ConfigProto(device_count={"CPU": cpu_num},  # limit to num_cpu_core CPU usage
                            inter_op_parallelism_threads=cpu_num,
                            intra_op_parallelism_threads=cpu_num)
    sess = tf.Session(config=config)

inputs = tf.placeholder(shape=[None, cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W, 3], dtype=tf.float32)
model = CenterNet(inputs, False)
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

hm = model.pred_hm
wh = model.pred_wh
reg = model.pred_reg
det = decode(hm, wh, reg, K=cfgs.SHOW_NUM)


class_names = read_class_names(cfgs.CLASS_FILE)

with open('./data/dataset/hand_test.txt', 'r') as f_read:
    txt_lines = f_read.readlines()

all_num = len(txt_lines)
correct_num = 0
for txt_line in txt_lines:

    txt = txt_line.split(' ')
    img_path = txt[0]
    obj = [float(i) for i in txt[1].split(',')]
    cls = int(obj[-1])
    points = obj[:4]

    original_image = cv2.imread(img_path)
    original_image_size = original_image.shape[:2]
    image_data = image_preprocess(np.copy(original_image), [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W])
    image_data = image_data[np.newaxis, ...]

    t0 = time.time()
    detections = sess.run(det, feed_dict={inputs: image_data})
    print('Inferencce took %.1f ms (%.2f fps)' % ((time.time() - t0) * 1000, 1 / (time.time() - t0)))
    detections = post_process(detections, original_image_size, [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W], cfgs.DOWN_RATIO,
                              cfgs.SCORE_THRESHOLD)

    if len(detections) == 0:
        if cls == -1:
            correct_num += 1
    elif cls >= 0 and cls == detections[0][-1] and cal_iou(detections[0], points) > cfgs.VAL_IOU_THRESH:
        correct_num += 1
print(correct_num / all_num)


