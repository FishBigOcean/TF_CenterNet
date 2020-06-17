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
from utils.utils import image_preprocess, py_nms, post_process, bboxes_draw_on_img, read_class_names

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
for txt_line in txt_lines:
    img_path = txt_line.split(' ')[0]
    img_point = txt_line.split(' ')[1:]

# img_names = os.listdir('D:/dataset/hand_network')
# for img_name in img_names:
#     img_path = 'D:/dataset/hand_network/' + img_name

    original_image = cv2.imread(img_path)
    original_image_size = original_image.shape[:2]
    image_data = image_preprocess(np.copy(original_image), [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W])
    image_data = image_data[np.newaxis, ...]

    t0 = time.time()
    detections = sess.run(det, feed_dict={inputs: image_data})
    print('Inferencce took %.1f ms (%.2f fps)' % ((time.time() - t0) * 1000, 1 / (time.time() - t0)))
    detections = post_process(detections, original_image_size, [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W],
                              cfgs.DOWN_RATIO,
                              cfgs.SCORE_THRESHOLD)
    if cfgs.USE_NMS:
        cls_in_img = list(set(detections[:, 5]))
        results = []
        for c in cls_in_img:
            cls_mask = (detections[:, 5] == c)
            classified_det = detections[cls_mask]
            classified_bboxes = classified_det[:, :4]
            classified_scores = classified_det[:, 4]
            inds = py_nms(classified_bboxes, classified_scores, max_boxes=50, iou_thresh=cfgs.NMS_THRESH)
            results.extend(classified_det[inds])
        results = np.asarray(results)
        if len(results) != 0:
            bboxes = results[:, 0:4]
            scores = results[:, 4]
            classes = results[:, 5]
            bboxes_draw_on_img(original_image, classes, scores, bboxes, class_names)
    else:
        bboxes = detections[:, 0:4]
        scores = detections[:, 4]
        classes = detections[:, 5]
        bboxes_draw_on_img(original_image, classes, scores, bboxes, class_names)

    cv2.imshow('img', original_image)
    cv2.waitKey()
