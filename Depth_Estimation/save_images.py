import scipy.io as spio
import numpy as np
import os
import cv2


def save_images():
    save_path = r"F:\Datasets\NYU Depth v2\Images"
    raw_depth_path = r"F:\Datasets\NYU Depth v2\rawDepths.mat"
    images_path = r"F:\Datasets\NYU Depth v2\images.mat"

    # depth maps
    mat = spio.loadmat(raw_depth_path, squeeze_me=True)
    depths = np.array(mat['rawDepths'])
    depths = depths.swapaxes(0, 2)
    depths = depths.swapaxes(1, 2)
    for i in range(len(depths)):
        max_pixel = np.amax(depths[i])
        for j in range(len(depths[i])):
            for k in range(len(depths[i][j])):
                depths[i][j][k] = int(depths[i][j][k] * 255 / max_pixel)
                # to handle any errors
                if depths[i][j][k] > 255:
                    depths[i][j][k] = 255
        if not os.path.exists(os.path.join(save_path, str(i+1))):
            os.makedirs(os.path.join(save_path, str(i+1)))
        path = os.path.join(save_path, str(i+1))
        cv2.imwrite(os.path.join(path, "depths{}.png".format(i+1)), depths[i])

    # rgb images
    mat = spio.loadmat(images_path, squeeze_me=True)
    images = np.array(mat['images'])
    images = images.swapaxes(0, 3)
    images = images.swapaxes(2, 3)
    images = images.swapaxes(1, 2)
    for i in range(len(images)):
        if i == 0:
            cv2.imshow('image 0', images[i])
            cv2.waitKey(0)
        if not os.path.exists(os.path.join(save_path, str(i+1))):
            os.makedirs(os.path.join(save_path, str(i+1)))
        path = os.path.join(save_path, str(i+1))
        cv2.imwrite(os.path.join(path, "images{}.png".format(i+1)), images[i])


def load_image():
    path = r"F:\Datasets\NYU Depth v2\Images\1\depths1.png"
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    cv2.imshow('This image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    save_images()
    # load_image()
