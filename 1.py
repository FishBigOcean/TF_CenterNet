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

# rom PIL import Image
# import imagehash, cv2, os


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

# import cv2
# import numpy as np
#
#
# def beauty_face(img):
#     '''
#     Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
#     https://my.oschina.net/wujux/blog/1563461
#     '''
#     # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
#     v1 = 5
#     v2 = 3
#     dx = v1 * 5  # 双边滤波参数之一
#     fc = v1 * 12.5  # 双边滤波参数之一
#     p = 0.1
#     temp1 = cv2.bilateralFilter(img, dx, fc, fc)
#     temp2 = cv2.subtract(temp1, img)
#     temp2 = cv2.add(temp2, (10, 10, 10, 128))
#     temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
#     temp4 = cv2.add(img, temp3)
#     dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
#     dst = cv2.add(dst, (10, 10, 10, 255))
#     return dst
#
#
# def beauty_face2(img):
#     '''
#     Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
#     '''
#     # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
#     v1 = 5
#     v2 = 3
#     dx = v1 * 5  # 双边滤波参数之一
#     fc = v1 * 12.5  # 双边滤波参数之一
#     p = 0.1
#     temp1 = cv2.bilateralFilter(img, dx, fc, fc)
#     temp2 = cv2.subtract(temp1, img)
#     temp2 = cv2.add(temp2, (10, 10, 10, 128))
#     temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
#     temp4 = cv2.subtract(cv2.add(cv2.add(temp3, temp3), img), (10, 10, 10, 255))
#     dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
#     dst = cv2.add(dst, (10, 10, 10, 255))
#     return dst
#
# img = cv2.imread('D:/dataset/hand_samples_sunzhi/new-all/process/all/new_1569_1.1_flip.jpg')
# dst = beauty_face(img)
# dst2 = beauty_face2(img)
# cv2.imshow("SRC", img)
# cv2.imshow("DST", dst)
# cv2.imshow('dst2', dst2)
#
# cv2.waitKey()


#
# import cv2, random, math
# import numpy as np
#
# def random_affine(img, targets=(), degrees=10, translate=.1, scale=0.3, shear=0, border=(0, 0)):
#     # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
#     # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
#     # targets = [cls, xyxy]
#
#     height = img.shape[0] + border[0] * 2  # shape(h,w,c)
#     width = img.shape[1] + border[1] * 2
#
#     # Rotation and Scale
#     R = np.eye(3)
#     a = random.uniform(-degrees, degrees)
#     # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
#     s = random.uniform(1 - scale, 1 + scale)
#     # s = 2 ** random.uniform(-scale, scale)
#     R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)
#
#     # Translation
#     T = np.eye(3)
#     T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border[1]  # x translation (pixels)
#     T[1, 2] = random.uniform(-translate, translate) * img.shape[0] + border[0]  # y translation (pixels)
#
#     # Shear
#     S = np.eye(3)
#     S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
#     S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
#
#     # Combined rotation matrix
#     M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
#     if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
#         img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
#
#     # Transform label coordinates
#     n = len(targets)
#     if n:
#         # cls x1, y1, x2, y2
#         # warp points
#         xy = np.ones((n * 4, 3))
#         xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
#         xy = (xy @ M.T)[:, :2].reshape(n, 8)
#
#         # xy2 = np.ones((n * 4, 3))
#         # xy2[:, :2] = np.sum(targets[:, [1, 2, 3, 4, 1, 4, 3, 4, 3, 2, 3, 2, 1, 2, 1, 4]].reshape(n * 4, 2, 2), axis=1) / 2
#         # xy2 = (xy2 @ M.T)[:, :2].reshape(n, 8)
#
#         # create new boxes
#         x = xy[:, [0, 2, 4, 6]]
#         y = xy[:, [1, 3, 5, 7]]
#         xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
#
#         # x2 = xy2[:, [0, 2, 4, 6]]
#         # y2 = xy2[:, [1, 3, 5, 7]]
#         # xy2 = np.concatenate((x2.min(1), y2.min(1), x2.max(1), y2.max(1))).reshape(4, n).T
#         # xy = xy2
#
#         # apply angle-based reduction of bounding boxes
#         radians = a * math.pi / 180
#         reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 16
#         x = (xy[:, 2] + xy[:, 0]) / 2
#         y = (xy[:, 3] + xy[:, 1]) / 2
#         w = (xy[:, 2] - xy[:, 0]) * reduction
#         h = (xy[:, 3] - xy[:, 1]) * reduction
#         xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
#
#         # reject warped points outside of image
#         xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
#         xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
#         w = xy[:, 2] - xy[:, 0]
#         h = xy[:, 3] - xy[:, 1]
#         area = w * h
#         area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
#         ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
#         i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)
#
#         targets = targets[i]
#         targets[:, 1:5] = xy[i]
#
#     return img, targets
#
# img = cv2.imread('D:/dataset/hand_samples_sunzhi/images-new/heart_zhubo_20171113000_265.jpg')
# boxes = np.array([[0, 373, 352, 608, 485]])
# img2, boxes2 = random_affine(img, boxes)
# boxes = boxes[0][1:]
# boxes2 = boxes2[0][1:]
# cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 3)
# cv2.rectangle(img2, (boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), (0, 255, 0), 3)
# cv2.imshow('img', img)
# cv2.imshow('img2', img2)
# cv2.waitKey()


# import numpy as np
# import math
# from PIL import Image
# import cv2, random
# import cfgs
#
#
# def load_mosaic():
#     # loads images in a mosaic
#     labels4 = []
#     s = cfgs.INPUT_IMAGE_W
#     yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in
#               [-cfgs.INPUT_IMAGE_H // 2, -cfgs.INPUT_IMAGE_W // 2]]  # mosaic center x, y
#     for i in range(4):
#         # Load image
#         img, bbox = load_random_img_label()
#         img, [bbox] = image_preprocess(np.copy(img), [cfgs.INPUT_IMAGE_H * 2, cfgs.INPUT_IMAGE_W * 2], np.copy([bbox]))
#         h, w, _ = img.shape
#         # place img in img4
#         if i == 0:  # top left
#             img4 = np.ones((s * 2, s * 2, img.shape[2]), dtype=np.uint8)  # base image with 4 tiles
#             means = np.array(cfgs.USED_MEANS_255[::-1], np.uint8).reshape((1, 1, 3))
#             print(means)
#             img4 = img4 * means
#             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
#             x1b, y1b, x2b, y2b = (w - (x2a - x1a)) // 2, (h - (y2a - y1a)) // 2, (w + (x2a - x1a)) // 2, (
#                         h + (y2a - y1a)) // 2  # xmin, ymin, xmax, ymax (small image)
#         elif i == 1:  # top right
#             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
#             x1b, y1b, x2b, y2b = (w - (x2a - x1a)) // 2, (h - (y2a - y1a)) // 2, (w + (x2a - x1a)) // 2, (
#                         h + (y2a - y1a)) // 2
#         elif i == 2:  # bottom left
#             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
#             x1b, y1b, x2b, y2b = (w - (x2a - x1a)) // 2, (h - (y2a - y1a)) // 2, (w + (x2a - x1a)) // 2, (
#                         h + (y2a - y1a)) // 2
#         elif i == 3:  # bottom right
#             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
#             x1b, y1b, x2b, y2b = (w - (x2a - x1a)) // 2, (h - (y2a - y1a)) // 2, (w + (x2a - x1a)) // 2, (
#                         h + (y2a - y1a)) // 2
#
#         img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
#         padw = x1a - x1b
#         padh = y1a - y1b
#
#         # Labels
#         labels = bbox.copy()
#         if len(bbox) > 0:  # Normalized xywh to pixel xyxy format
#             labels[0] += padw
#             labels[1] += padh
#             labels[2] += padw
#             labels[3] += padh
#         labels[[0, 2]] = np.clip(labels[[0, 2]], x1a, x2a)
#         labels[[1, 3]] = np.clip(labels[[1, 3]], y1a, y2a)
#         labels4.append([labels])
#         # Concat/clip labels
#     if len(labels4):
#         labels4 = np.concatenate(labels4, 0)
#         # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
#         np.clip(labels4[:, :4], 0, 2 * s, out=labels4[:, :4])  # use with random_affine
#     return img4, labels4
#
#
# def load_random_img_label():
#     with open(cfgs.TRAIN_DATA_FILE) as f_read:
#         read_lines = f_read.readlines()
#     length = len(read_lines)
#     index = random.randint(0, length - 1)
#     line = read_lines[index].split(' ')
#     img = cv2.imread(line[0])
#     bbox = [int(i) for i in line[1].split(',')]
#     return img, bbox
#
#
# def image_preprocess(image, target_size, gt_boxes=None):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#     ih, iw = target_size
#     h, w, _ = image.shape
#
#     scale = min(iw / w, ih / h)
#     nw, nh = int(scale * w), int(scale * h)
#     image_resized = cv2.resize(image, (nw, nh))
#     # image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=np.float32)
#     means = np.array(cfgs.USED_MEANS, np.float32).reshape((1, 1, 3))
#     std = np.array(cfgs.USED_STD, np.float32).reshape((1, 1, 3))
#     image_paded = np.ones((ih, iw, 3), dtype=np.float32)
#     image_paded = image_paded * means * 255
#     dw, dh = (iw - nw) // 2, (ih - nh) // 2
#     image_paded[dh: nh + dh, dw: nw + dw, :] = image_resized
#     image_paded = image_paded / 255.
#     image_paded = ((image_paded - means) / std).astype(np.float32)
#     image_paded = cv2.cvtColor(image_paded, cv2.COLOR_RGB2BGR).astype(np.float32)
#
#     if gt_boxes is None:
#         return image_paded
#     else:
#         gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
#         gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
#         return image_paded, gt_boxes
#
# img = cv2.imread('D:/dataset/hand_samples_sunzhi/new-all/process/image-v5-n/heart_plan_1510041135_75_1.0_flip.jpg')
# img2 = load_mosaic()
# cv2.imshow('img1', img)
# cv2.imshow('img2', img2)
# cv2.waitKey()

# import cv2 as cv
# import numpy as np
#
#
# def template_image():
#     target = cv.imread("D:/dataset/hand_samples_sunzhi/face.jpg", cv.IMREAD_GRAYSCALE)
#     tpl = cv.imread("D:/dataset/hand_samples_sunzhi/5_hand.jpg", cv.IMREAD_GRAYSCALE)
#     target =  cv.resize(target, (256, 256))
#     tpl = cv.resize(tpl, (256, 256))
#     #cv.imshow("modul", tpl)
#     #cv.imshow("yuan", target)
#     methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
#     th, tw = tpl.shape[:2]
#     for md in methods:
#         result = cv.matchTemplate(target, tpl, md)
#         min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
#         print(min_val, max_val)
#         if md == cv.TM_SQDIFF_NORMED:
#             tl = min_loc
#         else:
#             tl = max_loc
#         print(tl)
#         br = (tl[0] + tw, tl[1] + th)
#         cv.rectangle(target, tl, br, [0, 0, 0])
#         cv.imshow("pipei"+np.str(md), target)
#
#
# template_image()
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2
# import numpy as np
# import time
#
# def binaryMask(frame, x0, y0, width, height):
#     cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 画出截取的手势框图
#     roi = frame[y0:y0 + height, x0:x0 + width]  # 获取手势框图
#     start = time.time()
#     res = skinMask(roi)  # 进行肤色检测
#     print(time.time() - start)
#     return res


# def skinMask(roi):
#     YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
#     (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
#     cr1 = cv2.GaussianBlur(cr, (5,5), 0)
#     _, skin = cv2.threshold(cr1, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
#     kernel = np.ones((5,5),np.uint8)
#     for i in range(2):
#         skin = cv2.morphologyEx(skin, cv2.MORPH_ELLIPSE, kernel)
#     for i in range(4):
#         skin = cv2.morphologyEx(skin,cv2.MORPH_DILATE,kernel)
#     res = cv2.bitwise_and(roi,roi, mask = skin)
#     return res


# def skinMask(roi):
#     skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
#     cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1) #绘制椭圆弧线
#     YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
#     (y,Cr,Cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
#     (x,y) = Cr.shape
#     skinCrCbHist = np.array(skinCrCbHist)
#     cr = np.array(Cr, dtype=np.uint8).reshape([x * y, ])
#     cb = np.array(Cb, dtype=np.uint8).reshape([x * y, ])
#     skin = skinCrCbHist[cr, cb]
#     skin = np.clip(skin, 0, 1)
#     kernel = np.ones((5,5),np.uint8)
#     skin.resize([x, y])
#     for i in range(1):
#         skin = cv2.morphologyEx(skin, cv2.MORPH_ELLIPSE, kernel)
#     for i in range(4):
#         skin = cv2.morphologyEx(skin,cv2.MORPH_DILATE,kernel)
#     res = roi * np.expand_dims(skin, -1)
#     return res


# def skinMask(roi):
#     YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
#     (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
#     skin = np.zeros(cr.shape, dtype = np.uint8)
#     (x,y) = cr.shape
#     for i in range(0, x):
#         for j in range(0, y):
#             #每个像素点进行判断
#             if(cr[i][j] > 130) and (cr[i][j] < 175) and (cb[i][j] > 77) and (cb[i][j] < 127):
#                 skin[i][j] = 255
#     res = cv2.bitwise_and(roi,roi, mask = skin)
#     return res
#
#
#
# def beauty_face2(img):
#     '''
#     Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
#     '''
#     # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
#     v1 = 5
#     v2 = 3
#     dx = v1 * 5  # 双边滤波参数之一
#     fc = v1 * 12.5  # 双边滤波参数之一
#     p = 0.1
#     temp1 = cv2.bilateralFilter(img, dx, fc, fc)
#     temp2 = cv2.subtract(temp1, img)
#     temp2 = cv2.add(temp2, (10, 10, 10, 128))
#     temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
#     temp4 = cv2.subtract(cv2.add(cv2.add(temp3, temp3), img), (10, 10, 10, 255))
#     dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
#     dst = cv2.add(dst, (10, 10, 10, 255))
#     return dst
#
# def beauty_face(img):
#     '''
#     Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
#     https://my.oschina.net/wujux/blog/1563461
#     '''
#     # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
#     v1 = 3
#     v2 = 1
#     dx = v1 * 5  # 双边滤波参数之一
#     fc = v1 * 12.5  # 双边滤波参数之一
#     p = 0.1
#     temp1 = cv2.bilateralFilter(img, dx, fc, fc)
#     temp2 = cv2.subtract(temp1, img)
#     temp2 = cv2.add(temp2, (10, 10, 10, 128))
#     temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
#     temp4 = cv2.add(img, temp3)
#     dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
#     dst = cv2.add(dst, (10, 10, 10, 255))
#     return dst
#
# for i in range(10):
#     img = cv2.imread('D:/dataset/hand_samples_sunzhi/c%d.jpg' % (i + 1))
#     img = cv2.resize(img, (256, 256))
#     mask = binaryMask(img, 0, 0, img.shape[1], img.shape[0])


# def getContours(img):
#     kernel = np.ones((5,5),np.uint8)
#     closed = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
#     closed = cv2.morphologyEx(closed,cv2.MORPH_CLOSE,kernel)
#     _,contours,h = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     vaildContours = []
#     for cont in contours:
#         if cv2.contourArea(cont)>9000:
#             #x,y,w,h = cv2.boundingRect(cont)
#             #if h/w >0.75:
#             #filter face failed
#             vaildContours.append(cv2.convexHull(cont))
#             #rect = cv2.minAreaRect(cont)
#             #box = cv2.cv.BoxPoint(rect)
#             #vaildContours.append(np.int0(box))
#     return  vaildContours
#
# cap = cv2.VideoCapture('D:/dataset/hand_samples_sunzhi/c6.jpg')
# while(cap.isOpened()):
#     ret,img = cap.read()
#     skin = skinMask(img)
#     contours = getContours(skin)
#     cv2.drawContours(img,contours,-1,(0,255,0),2)
#     cv2.imshow('capture',img)
#     k = cv2.waitKey()
#     if k == 27:
#         break


import tensorflow as tf
import matplotlib.pyplot as plt
import cfgs
style1 = []
style2 = []
N = 400

with tf.Session() as sess:
    global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
    warmup_steps = tf.constant(10, dtype=tf.float32, name='warmup_steps')
    learing_rate1 = tf.cond(
        pred=global_step < warmup_steps,
        true_fn=lambda: global_step / warmup_steps * cfgs.INIT_LR_CR,
        false_fn=lambda: tf.train.cosine_decay_restarts(cfgs.INIT_LR_CR, global_step, 180, cfgs.T_MUL,
                                                        cfgs.M_MUL)
    )
    sess.run(tf.global_variables_initializer())
    learing_rate2 = tf.train.cosine_decay(
        learning_rate=0.001, global_step=global_step, decay_steps=50)
    global_step_op = tf.assign_add(global_step, 1.0)
    for step in range(N):
        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])
        temp = sess.run([global_step_op])
        style1.append(lr1)
        style2.append(lr2)

step = range(N)

plt.plot(step, style1, 'g-', linewidth=2, label='cosine_decay_restarts')
plt.plot(step, style2, 'r--', linewidth=1, label='cosine_decay')
plt.title('cosine_decay_restarts')
plt.xlabel('step')
plt.ylabel('learing rate')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
