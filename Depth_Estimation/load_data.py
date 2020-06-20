import scipy.io as spio
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_data():
    print("Retrieving Data")
    raw_depth_path = r"F:\Datasets\NYU Depth v2\rawDepths.mat"
    mat = spio.loadmat(raw_depth_path, squeeze_me=True)

    depths = np.array(mat['rawDepths'])
    depths = depths.swapaxes(0, 2)
    depths = cv2.resize(depths, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
    print(np.shape(depths))
    # depths = depths.swapaxes(1, 2)
    # print(np.shape(depths))

    images_path = r"F:\Datasets\NYU Depth v2\images.mat"
    mat = spio.loadmat(images_path, squeeze_me=True)

    images = np.array(mat['images'])
    images = images.swapaxes(0, 3)
    images = images.swapaxes(1, 2)
    images = cv2.resize(images, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
    print(np.shape(images))
    # print(np.shape(images))

    # print(images[0][0][0])
    # print(len(images[0][0][0]))
    # print(len(images[0][0]))
    # print(images[0])

    num_images = len(images)
    data = []
    # print("Loop Starting")
    for i in range(num_images):
        # print(i)
        # print(np.shape(images[i][0]), np.shape(images[i][1]), np.shape(images[i][2]), np.shape(depths[i]))
        data.append(np.concatenate(([images[i][0]], [images[i][1]], [images[i][2]], [depths[i]]), axis=0))
    # print("Loop Ending")

    # print(len(data))
    # print(len(data[0]))
    # print(len(data[0][0]))
    # print(len(data[0][0][0]))
    print("Data Retrieved")

    return data


def load_data():
    path = r"F:\Datasets\NYU Depth v2\Images"
    data = []
    for directory in tqdm(os.listdir(path)):
        current_dir = os.path.join(path, directory)
        temp = []
        for img in os.listdir(current_dir):
            image = os.path.join(current_dir, img)
            if 'depths' in img:
                temp.append(cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE), (150, 150)))
            else:
                temp.append(cv2.resize(cv2.imread(image), (150, 150)))
        combined = np.dstack((temp[1], temp[0]))
        combined = combined.swapaxes(0, 2)
        data.append(combined)

    print("Shape of data is:", np.shape(data))
    return data


if __name__ == '__main__':
    # data_path = r"F:\Datasets\NYU Depth v2\images.mat"
    # mat = spio.loadmat(data_path, squeeze_me=True)
    #
    # # print(mat['depths'])
    #
    # depths = np.array(mat['images'])
    # print(np.shape(depths))
    #
    # depths = depths.swapaxes(0, 3)
    # # depths = depths.swapaxes(2, 3)
    # depths = depths.swapaxes(1, 2)
    #
    # # depths = depths.reshape(1449, 640, 480)
    #
    # print(np.shape(depths))
    #
    # print(np.shape(depths[0]))
    #
    # # print(depths[0])
    # print(len(depths))
    #
    # cv2.imshow('Image1', depths[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # temp = np.uint8(depths[0])
    # colorDepth = cv2.applyColorMap(temp, cv2.COLORMAP_HSV)
    # cv2.imshow('Color Depth', colorDepth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # plt.imshow(depths[0], cmap="plasma")
    # plt.show()

    load_data()
