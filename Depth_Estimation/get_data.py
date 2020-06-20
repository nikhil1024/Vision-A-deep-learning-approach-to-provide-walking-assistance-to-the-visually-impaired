import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle


def get_data():
    dataset_dir = r"F:\Datasets\KITTI drive"
    data = []

    for root_dir in tqdm(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, root_dir)
        for sub_dir in os.listdir(path):
            count = 0
            count_left, count_right = 0000000000, 0000000000
            path = os.path.join(path, sub_dir)
            left_image_path = os.path.join(path, "image_02", "data")
            right_image_path = os.path.join(path, "image_03", "data")

            for _ in range(len(os.listdir(left_image_path))):  # 2*len() to retrieve all the images. But currently I only want to retrieve half of the images
                if count % 2 == 0:
                    left_image = cv2.resize(cv2.imread(os.path.join(left_image_path, str(count_left).zfill(10) + ".png"), cv2.IMREAD_GRAYSCALE), (150, 150))
                    count_left += 1
                else:
                    right_image = cv2.resize(cv2.imread(os.path.join(right_image_path, str(count_right).zfill(10) + ".png"), cv2.IMREAD_GRAYSCALE), (150, 150))
                    count_right += 1

                    data.append([np.array(left_image), np.array(right_image)])

                count += 1

    # shuffle(data)

    train_set = data[:-int(len(data)*0.3)]
    test_set = data[-int(len(data)*0.3):]
    train_x, train_y, test_x, test_y = [], [], [], []

    for sample in range(len(train_set)):
        train_x.append(train_set[sample][0])
        train_y.append(train_set[sample][1])

    for sample in range(len(test_set)):
        test_x.append(test_set[sample][0])
        test_y.append(test_set[sample][1])

    return train_x, train_y, test_x, test_y
