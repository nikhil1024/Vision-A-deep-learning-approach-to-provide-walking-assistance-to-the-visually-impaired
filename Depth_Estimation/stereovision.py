import cv2
import numpy as np
import os


def calibrate_camera():
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS, cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # preparing object points
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # array to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    image_path = r""  # calibration images here

    for img_name in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, img_name), cv2.IMREAD_GRAYSCALE)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(img, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)


def calculate_disparity():
    pass


def calculate_depth():
    pass
