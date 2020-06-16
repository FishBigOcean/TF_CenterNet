import os
import argparse

def convert_voc_annotation(data_path, train_path, test_path):

    classes = ['love', 'like', 'scissors']
    img_list = os.listdir(data_path + '/relabels-new')
    train_num = 0
    test_num = 0
    with open(train_path, 'w') as f_train:
        with open(test_path, 'w') as f_test:
            for i, image_ind in enumerate(img_list):
                image_path = data_path + '/images-new/' + image_ind.replace('txt', 'jpg')
                annotation = image_path
                label_path = data_path + '/relabels-new/' + image_ind
                with open(label_path, 'r') as f_read:
                    txt = f_read.readlines()[0].split(' ')
                    class_ind, x_min, y_min, x_max, y_max = txt
                    annotation += ' ' + ','.join([x_min, y_min, x_max, y_max, class_ind])
                print(annotation)
                if i % 500 < 50:
                    f_test.write(annotation + "\n")
                    test_num += 1
                else:
                    f_train.write(annotation + '\n')
                    train_num += 1
    return train_num, test_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="D:/dataset/hand_samples_sunzhi")
    parser.add_argument("--train_annotation", default="./hand_train_new.txt")
    parser.add_argument("--test_annotation",  default="./hand_test_new.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    num_train, num_test = convert_voc_annotation(flags.data_path, flags.train_annotation, flags.test_annotation)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' % (num_train, num_test))
