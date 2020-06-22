# import cv2
# import numpy as np
# import math
# import matplotlib.pyplot as plt
#
# img1 = cv2.imread('D:/dataset/hand_samples_sunzhi/images-new/neg_a_1360.jpg')
#
#
# def gammaTranform(gamma, image):
#     # h, w, d = image.shape[0], image.shape[1], image.shape[2]
#     # image = image / 255.0
#     # new_img = np.zeros((h, w, d), dtype=np.float32)
#     # for i in range(h):
#     #     for j in range(w):
#     #         new_img[i, j, 0] = c * math.pow(image[i, j, 0], gamma)
#     #         new_img[i, j, 1] = c * math.pow(image[i, j, 1], gamma)
#     #         new_img[i, j, 2] = c * math.pow(image[i, j, 2], gamma)
#     # cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
#     # new_img = cv2.convertScaleAbs(new_img)
#     img_norm = image / 255.0
#     img_gamma = np.power(img_norm, gamma) * 255.0
#     img_gamma = img_gamma.astype(np.uint8)
#     return img_gamma
#
#
# def equalizeHist(img):
#     (b, g, r) = cv2.split(img)
#     bH = cv2.equalizeHist(b)
#     gH = cv2.equalizeHist(g)
#     rH = cv2.equalizeHist(r)
#     result = cv2.merge((bH, gH, rH))
#     return result
#
#
# img2 = gammaTranform(0.6, img1)
# clahe = cv2.createCLAHE(3, (8, 8))
# img3 = clahe.apply(img1)
# print(np.max(img1), np.max(img2), np.max(img3))
#
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)
# cv2.waitKey()

#
# import cv2, numpy, os
# input_dir = 'D:/dataset/hand_samples_sunzhi/'
# img_dir = input_dir + 'images-select'
# txt_dir = input_dir + 'relabels-new'
# out_dir0 = input_dir + 'new/0'
# out_dir1 = input_dir + 'new/1'
# out_dir2 = input_dir + 'new/2'
# out_dir_list = [out_dir0, out_dir1, out_dir2]
# txt_list = os.listdir(txt_dir)
# txt_set = set(txt_list)
# img_list = os.listdir(img_dir)
# num = 0
# for img in img_list:
#     txt = img.replace('jpg', 'txt')
#     if txt in txt_set:
#         num += 1
#         print(txt)
#         with open(txt_dir + '/' + txt) as f_read:
#             cls = int(f_read.readlines()[0][0])
#             im = cv2.imread(img_dir + '/' + img)
#             cv2.imwrite(out_dir_list[cls] + '/' + img, im)
# print(num)

from PIL import Image
import imagehash, cv2, os


# def cal_hamming(a, b):
#     # compute and return the Hamming distance between the integers
#     return bin(int(a, 16) ^ int(b, 16)).count("1")
#
#
# def cal_hash(img_dir):
#     hash = str(imagehash.phash(Image.open('D:/dataset/hand_samples_sunzhi/del/flip/' + img_dir)))
#     return hash
#
#
# img_list = os.listdir('D:/dataset/hand_samples_sunzhi/del/flip')
# hash_list = []
# num = 0
# for img_dir in img_list:
#     cur_val = cal_hash(img_dir)
#     for pre_val in hash_list:
#         if cal_hamming(cur_val, pre_val) <= 10:
#             break
#     else:
#         num += 1
#         img = cv2.imread('D:/dataset/hand_samples_sunzhi/del/flip/' + img_dir)
#         cv2.imwrite('D:/dataset/hand_samples_sunzhi/del/flip_save/' + img_dir, img)
#         hash_list.append(cur_val)
#         print(img_dir)
#
# print(num)


# import cv2
# import numpy as np
# import os
#
#
# def gamma_trans(img, gamma):  # gamma函数处理
#     gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
#     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
#     return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。
#
#
# def nothing(x):
#     pass
#
#
# cv2.namedWindow("demo", 0)  # 将显示窗口的大小适应于显示器的分辨率
# cv2.createTrackbar('Value of Gamma', 'demo', 100, 1000, nothing)  # 使用滑动条动态调节参数gamma
#
# data_base_dir = "D:/dataset/hand_samples_sunzhi/new-all/all"  # 输入文件夹的路径
# outfile_dir = "D:/dataset/hand_samples_sunzhi/new-all/all-process"  # 输出文件夹的路径
# processed_number = 0  # 统计处理图片的数量
# print("press enter to make sure your operation and process the next picture")
#
# for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
#     read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
#     image = cv2.imread(read_img_name)  # 读入图片
#
#     while (1):
#         value_of_gamma = cv2.getTrackbarPos('Value of Gamma', 'demo')  # gamma取值
#         value_of_gamma = value_of_gamma * 0.01  # 压缩gamma范围，以进行精细调整
#         image_gamma_correct = gamma_trans(image, value_of_gamma)  # 2.5为gamma函数的指数值，大于1曝光度下降，大于0小于1曝光度增强
#         cv2.imshow("demo", image_gamma_correct)
#         k = cv2.waitKey(1)
#         if k == 13:  # 按回车键确认处理、保存图片到输出文件夹和读取下一张图片
#             processed_number += 1
#             out_img_name = outfile_dir + '//' + file.strip()
#             cv2.imwrite(out_img_name, image_gamma_correct)
#             print("The number of photos which were processed is ", processed_number)
#             break




