import os
import numpy as np
import tensorflow as tf
import cv2
import math
import time
import shutil
import cfgs
from tqdm import tqdm
from CenterNet import CenterNet
from utils.generator import get_data
from net.resnet import load_weights
import tensorflow.contrib.slim as slim


def train():
    # define dataset
    num_train_imgs = len(open(cfgs.TRAIN_DATA_FILE, 'r').readlines())
    num_train_batch = int(math.ceil(float(num_train_imgs) / cfgs.BATCH_SIZE))
    num_test_imgs = len(open(cfgs.TEST_DATA_FILE, 'r').readlines())
    num_test_batch = int(math.ceil(float(num_test_imgs) / 4))

    # train dataset
    train_dataset = tf.data.TextLineDataset(cfgs.TRAIN_DATA_FILE)
    train_dataset = train_dataset.shuffle(num_train_imgs)
    train_dataset = train_dataset.batch(cfgs.BATCH_SIZE)
    train_dataset = train_dataset.map(lambda x: tf.py_func(get_data, inp=[x, cfgs.USE_AUG],
                                                           Tout=[tf.float32, tf.float32, tf.float32, tf.float32,
                                                                 tf.float32, tf.float32, tf.float32]),
                                      num_parallel_calls=8)
    train_dataset = train_dataset.prefetch(8)

    # test dataset
    test_dataset = tf.data.TextLineDataset(cfgs.TEST_DATA_FILE)
    test_dataset = test_dataset.batch(4)
    test_dataset = test_dataset.map(lambda x: tf.py_func(get_data, inp=[x, False],
                                                         Tout=[tf.float32, tf.float32, tf.float32, tf.float32,
                                                               tf.float32, tf.float32, tf.float32]),
                                    num_parallel_calls=8)
    test_dataset = test_dataset.prefetch(8)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    trainset_init_op = iterator.make_initializer(train_dataset)
    testset_init_op = iterator.make_initializer(test_dataset)

    input_data, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind, batch_cls = iterator.get_next()
    input_data.set_shape([cfgs.BATCH_SIZE, cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W, 3])
    batch_hm.set_shape(
        [cfgs.BATCH_SIZE, cfgs.INPUT_IMAGE_H // cfgs.DOWN_RATIO, cfgs.INPUT_IMAGE_W // cfgs.DOWN_RATIO, cfgs.NUM_CLASS])
    batch_wh.set_shape([cfgs.BATCH_SIZE, cfgs.MAX_OBJ, 2])
    batch_reg.set_shape([cfgs.BATCH_SIZE, cfgs.MAX_OBJ, 2])
    batch_reg_mask.set_shape([cfgs.BATCH_SIZE, cfgs.MAX_OBJ])
    batch_ind.set_shape([cfgs.BATCH_SIZE, cfgs.MAX_OBJ])
    batch_cls.set_shape([cfgs.BATCH_SIZE, cfgs.MAX_OBJ])

    # training flag
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    # define model and loss
    model = CenterNet(input_data, is_training)
    with tf.variable_scope('loss'):
        hm_loss, wh_loss, reg_loss, cls_loss = model.compute_loss(batch_hm, batch_wh, batch_reg, batch_reg_mask,
                                                                  batch_ind, batch_cls)
        regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = hm_loss + wh_loss + reg_loss + cls_loss + regular_loss

    # define train op
    if cfgs.LR_TPYE == "CosineAnnealing":
        global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        warmup_steps = tf.constant(cfgs.WARM_UP_EPOCHS * num_train_batch, dtype=tf.float64, name='warmup_steps')
        train_steps = tf.constant(cfgs.EPOCHS * num_train_batch, dtype=tf.float64, name='train_steps')
        learning_rate = tf.cond(
            pred=global_step < warmup_steps,
            true_fn=lambda: global_step / warmup_steps * cfgs.INIT_LR,
            false_fn=lambda: cfgs.END_LR + 0.5 * (cfgs.INIT_LR - cfgs.END_LR) *
                             (1 + tf.cos(
                                 (global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
        )
        global_step_update = tf.assign_add(global_step, 1.0)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([optimizer, global_step_update]):
                train_op = tf.no_op()

    else:
        global_step = tf.Variable(0, trainable=False)
        if cfgs.LR_TPYE == "exponential":
            learning_rate = tf.train.exponential_decay(cfgs.LR,
                                                       global_step,
                                                       cfgs.LR_DECAY_STEPS,
                                                       cfgs.LR_DECAY_RATE,
                                                       staircase=True)
        elif cfgs.LR_TPYE == "piecewise":
            learning_rate = tf.train.piecewise_constant(global_step, cfgs.LR_BOUNDARIES, cfgs.LR_PIECEWISE)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfgs.MAX_KEEP)
    pre_test_loss = float('inf')

    with tf.Session() as sess:
        with tf.name_scope('summary'):
            tf.summary.scalar("learning_rate", learning_rate)
            tf.summary.scalar("hm_loss", hm_loss)
            tf.summary.scalar("wh_loss", wh_loss)
            tf.summary.scalar("reg_loss", reg_loss)
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('regular_loss', regular_loss)
            tf.summary.scalar("total_loss", total_loss)

            logdir = './log/' + cfgs.VERSION
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            write_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

            ckptdir = './checkpoint/' + cfgs.VERSION
            if os.path.exists(ckptdir): shutil.rmtree(ckptdir)
            os.mkdir(ckptdir)

        # train 
        sess.run(tf.global_variables_initializer())
        if cfgs.PRE_TRAIN:
            load_weights(sess, './pretrained_weights/mobilenet_v2.npy')
        for epoch in range(1, 1 + cfgs.EPOCHS):
            pbar = tqdm(range(num_train_batch))
            train_epoch_loss, test_epoch_loss = [], []
            train_hm_loss, train_wh_loss, train_reg_loss, train_cls_loss, train_regular_loss = [], [], [], [], []
            sess.run(trainset_init_op)
            for i in pbar:
                _, summary, train_step_loss, global_step_val, _hm_loss, _wh_loss, _reg_loss, _cls_loss, _regular_loss = sess.run(
                    [train_op, write_op, total_loss, global_step, hm_loss, wh_loss, reg_loss, cls_loss, regular_loss],
                    feed_dict={is_training: True})

                train_epoch_loss.append(train_step_loss)
                train_hm_loss.append(_hm_loss)
                train_wh_loss.append(_wh_loss)
                train_reg_loss.append(_reg_loss)
                train_cls_loss.append(_cls_loss)
                train_regular_loss.append(_regular_loss)
                summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            sess.run(testset_init_op)
            for j in range(num_test_batch):
                test_step_loss = sess.run(total_loss, feed_dict={is_training: False})
                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            train_hm_loss, train_wh_loss, train_reg_loss, train_cls_loss, train_regular_loss = np.mean(
                train_hm_loss), np.mean(train_wh_loss), np.mean(train_reg_loss), np.mean(train_cls_loss), np.mean(
                train_regular_loss)
            ckpt_file = ckptdir + "/centernet_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f"
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss))
            print('hm loss: %.4f  wh loss: %.4f  reg loss: %.4f  cls loss: %.4f  regular loss: %.4f' % (
                train_hm_loss, train_wh_loss, train_reg_loss, train_cls_loss, train_regular_loss))
            if cfgs.SAVE_MIN:
                if test_epoch_loss < pre_test_loss:
                    pre_test_loss = test_epoch_loss
                    print('Saving  %s' % (ckpt_file))
                    saver.save(sess, ckpt_file, global_step=epoch)
            else:
                saver.save(sess, ckpt_file, global_step=epoch)

if __name__ == '__main__': train()
