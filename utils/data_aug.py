import random
import numpy as np
import cv2, math
from PIL import Image
import cfgs

def random_grid_mask(image, rotate=23, ratio=0.5, prob=1.):
    if np.random.rand() > prob:
        return image
    image_shape = image.shape
    h, w = image_shape[0], image_shape[1]
    d1, d2 = min(image_shape[:2]) // 6, min(image_shape[:2]) // 4

    # 1.5 * h, 1.5 * w works fine with the squared images
    # But with rectangular input, the mask might not be able to recover back to the input image shape
    # A square mask with edge length equal to the diagnoal of the input image
    # will be able to cover all the image spot after the rotation. This is also the minimum square.
    hh = math.ceil((math.sqrt(h * h + w * w)))

    d = np.random.randint(d1, d2)
    # d = self.d

    # maybe use ceil? but i guess no big difference
    l = math.ceil(d * ratio)

    mask = np.ones((hh, hh), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    for i in range(-1, hh // d + 1):
        s = d * i + st_h
        t = s + l
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        mask[s:t, :] *= 0
    for i in range(-1, hh // d + 1):
        s = d * i + st_w
        t = s + l
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        mask[:, s:t] *= 0
    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
    mask = 1 - mask
    if image_shape[-1] == 3:
        mask = np.stack([mask, mask, mask], axis=-1)
    image = image * mask
    return image


def random_gamma_tranform(image):
    if random.random() < 0.5:
        gamma_list = [0.9, 0.95, 1.1]
        gamma = gamma_list[random.randint(0, 2)]
        image = image / 255.0
        image = np.power(image, gamma) * 255.0
        image = image.astype(np.uint8)
    return image


def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - 1 - bboxes[:, [2, 0]]

    return image, bboxes


def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans * 2 // 3) - max_l_trans // 3))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans * 2 // 3) - max_u_trans // 3))
        crop_xmax = min(w, int(max_bbox[2] + random.uniform(0, max_r_trans * 2 // 3) + max_r_trans // 3))
        crop_ymax = min(h, int(max_bbox[3] + random.uniform(0, max_d_trans * 2 // 3) + max_d_trans // 3))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes


def random_affine(image, bboxes=(), degrees=10, translate=0.05, scale=0.3, shear=0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(image.shape[1] / 2, image.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * image.shape[1] + border[1]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * image.shape[0] + border[0]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        image = cv2.warpAffine(image, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=tuple(cfgs.USED_MEANS_255[::-1]))

    # Transform label coordinates
    n = len(bboxes)
    if n:
        # cls x1, y1, x2, y2
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # xy2 = np.ones((n * 4, 3))
        # xy2[:, :2] = np.sum(targets[:, [1, 2, 3, 4, 1, 4, 3, 4, 3, 2, 3, 2, 1, 2, 1, 4]].reshape(n * 4, 2, 2), axis=1) / 2
        # xy2 = (xy2 @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # x2 = xy2[:, [0, 2, 4, 6]]
        # y2 = xy2[:, [1, 3, 5, 7]]
        # xy2 = np.concatenate((x2.min(1), y2.min(1), x2.max(1), y2.max(1))).reshape(4, n).T
        # xy = xy2

        # apply angle-based reduction of bounding boxes
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 16
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

        bboxes = bboxes[i]
        bboxes[:, :4] = xy[i]

    return image, bboxes


def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - 1 - max_bbox[2]
        max_d_trans = h - 1 - max_bbox[3]

        tx = random.uniform(-(max_l_trans // 3), (max_r_trans // 3))
        ty = random.uniform(-(max_u_trans // 3), (max_d_trans // 3))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes


def random_beauty_face(img):
    def beauty_face(img):
        '''
        Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
        https://my.oschina.net/wujux/blog/1563461
        '''
        # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
        v1 = 5
        v2 = 3
        dx = v1 * 5  # 双边滤波参数之一
        fc = v1 * 12.5  # 双边滤波参数之一
        p = 0.1
        temp1 = cv2.bilateralFilter(img, dx, fc, fc)
        temp2 = cv2.subtract(temp1, img)
        temp2 = cv2.add(temp2, (10, 10, 10, 128))
        temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
        temp4 = cv2.add(img, temp3)
        dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
        dst = cv2.add(dst, (10, 10, 10, 255))
        return dst

    def beauty_face2(img):
        '''
        Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
        '''
        # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
        v1 = 5
        v2 = 3
        dx = v1 * 5  # 双边滤波参数之一
        fc = v1 * 12.5  # 双边滤波参数之一
        p = 0.1
        temp1 = cv2.bilateralFilter(img, dx, fc, fc)
        temp2 = cv2.subtract(temp1, img)
        temp2 = cv2.add(temp2, (10, 10, 10, 128))
        temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
        temp4 = cv2.subtract(cv2.add(cv2.add(temp3, temp3), img), (10, 10, 10, 255))
        dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
        dst = cv2.add(dst, (10, 10, 10, 255))
        return dst

    if random.random() < 0.75:
        if random.random() < 0.5:
            img = beauty_face(img)
        else:
            img = beauty_face2(img)
    return img


def random_hsv(img, hgain=0.014, sgain=0.68, vgain=0.36):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''

    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(- 0.6 * val_vari, 1.2 * val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def skinMask(img):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, Cr, Cb) = cv2.split(YCrCb)
    (x, y) = Cr.shape
    skinCrCbHist = np.array(skinCrCbHist)
    cr = np.array(Cr, dtype=np.uint8).reshape([x * y, ])
    cb = np.array(Cb, dtype=np.uint8).reshape([x * y, ])
    skin = skinCrCbHist[cr, cb]
    skin = np.clip(skin, 0, 1)
    kernel = np.ones((5, 5), np.uint8)
    skin.resize([x, y])
    for i in range(1):
        skin = cv2.morphologyEx(skin, cv2.MORPH_ELLIPSE, kernel)
    for i in range(4):
        skin = cv2.morphologyEx(skin, cv2.MORPH_DILATE, kernel)
    res = img * np.expand_dims(skin, -1)
    return res


def load_mosaic(img, bbox):
    # loads images in a mosaic
    labels4 = []
    s = cfgs.INPUT_IMAGE_W // 2
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in
              [-cfgs.INPUT_IMAGE_H // 4, -cfgs.INPUT_IMAGE_W // 4]]  # mosaic center x, y
    for i in range(4):
        # place img in img4
        if i == 0:  # top left
            img4 = np.ones((s * 2, s * 2, img.shape[2]), dtype=np.float32)
            means = np.array(cfgs.USED_MEANS_255[::-1], np.float32).reshape((1, 1, 3))
            img4 = img4 * means
            img4 = img4.astype(np.uint8)
            img, [bbox] = image_preprocess_mosaic(np.copy(img), [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W],
                                                  np.copy(bbox))
            h, w, _ = img.shape
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = (w - (x2a - x1a)) // 2, (h - (y2a - y1a)) // 2, (w + (x2a - x1a)) // 2, (
                    h + (y2a - y1a)) // 2  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            img, bbox = load_random_img_label()
            img, [bbox] = image_preprocess_mosaic(np.copy(img), [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W],
                                                  np.copy([bbox]))
            h, w, _ = img.shape
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = (w - (x2a - x1a)) // 2, (h - (y2a - y1a)) // 2, (w + (x2a - x1a)) // 2, (
                    h + (y2a - y1a)) // 2
        elif i == 2:  # bottom left
            img, bbox = load_random_img_label()
            img, [bbox] = image_preprocess_mosaic(np.copy(img), [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W],
                                                  np.copy([bbox]))
            h, w, _ = img.shape
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = (w - (x2a - x1a)) // 2, (h - (y2a - y1a)) // 2, (w + (x2a - x1a)) // 2, (
                    h + (y2a - y1a)) // 2
        elif i == 3:  # bottom right
            img, bbox = load_random_img_label()
            img, [bbox] = image_preprocess_mosaic(np.copy(img), [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W],
                                                  np.copy([bbox]))
            h, w, _ = img.shape
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = (w - (x2a - x1a)) // 2, (h - (y2a - y1a)) // 2, (w + (x2a - x1a)) // 2, (
                    h + (y2a - y1a)) // 2

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels = bbox.copy()
        if len(bbox) > 0:  # Normalized xywh to pixel xyxy format
            labels[0] += padw
            labels[1] += padh
            labels[2] += padw
            labels[3] += padh
        labels[[0, 2]] = np.clip(labels[[0, 2]], x1a, x2a)
        labels[[1, 3]] = np.clip(labels[[1, 3]], y1a, y2a)
        if (labels[2] - labels[0]) * (labels[3] - labels[1]) < 0.5 * (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]):
            labels[-1] = cfgs.NUM_CLASS
        labels4.append(labels)
    labels4 = np.array(labels4)
    return img4, labels4


def load_random_img_label():
    with open(cfgs.TRAIN_DATA_FILE) as f_read:
        read_lines = f_read.readlines()
    length = len(read_lines)
    index = random.randint(0, length - 1)
    line = read_lines[index].split(' ')
    img = cv2.imread(line[0])
    bbox = [int(i) for i in line[1].split(',')]
    return img, bbox


def image_preprocess_mosaic(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    means = np.array(cfgs.USED_MEANS_255[::-1], np.float32).reshape((1, 1, 3))
    image_paded = np.ones((ih, iw, 3), dtype=np.float32)
    image_paded = image_paded * means
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh: nh + dh, dw: nw + dw, :] = image_resized

    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
