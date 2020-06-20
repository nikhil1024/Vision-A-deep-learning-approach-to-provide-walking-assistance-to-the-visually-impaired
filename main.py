import threading
import cv2
import time
from Text_to_Speech import speak
from Object_Detection import object_detector
from Stereo_Vision import stereo_vision


def to_object_detection(img):
    global buffer, obj, pos
    # while buffer > 2 or len(obj) > 1:

    #     time.sleep(1)

    new_text, new_coords, new_pos = object_detector.detect_objects(img)
    obj.append(new_text)
    coords.append(new_coords)
    pos.append(new_pos)
    print('\n\nObject: ', obj, '\nPos: ', pos[-1], "co ", coords)
    # buffer += 1


def to_depth_estimation(left, right):
    global buffer, depth
    # while buffer < -2:
    #     time.sleep(1)

    new_depth = stereo_vision.generate_depth_map(left, right)
    depth.append(new_depth)
    # buffer -= 1


def sum_list(l, a, b, c, d):
    l1 = []
    if a < 0:
        a = 0
    if a + c > 400:
        c = 400 - a - 1
    if b < 0:
        b = 0
    if b + d > 300:
        d = 300 - b - 1

    # print('\n\n\n', a, b, c, d)
    for x in l[int(b):int(b)+int(d)]:
        for y in x[int(a):int(a)+int(c)]:
            l1.append(y)
            # print(y)
    # print('sum 12121', sum(l1))
    return sum(l1)


def to_speech():
    global obj, depth, pos
    # while len(obj) > 1 and depth > 1:
    #     time.sleep(1)
    if len(obj) == 0:
        return
    current_obj = obj.pop(0)
    current_depth = depth.pop(0)
    current_coords = coords.pop(0)
    current_pos = pos.pop(0)
    # for x in current_coords:
    #     print(sum_list(current_depth, x[0], x[2], x[1], x[3]))
    #     print(sum(sum(current_depth)))
    #     if sum(sum(current_depth) < sum_list(current_depth, x[0], x[2], x[1], x[3])):
    #         print('Near')
    #     else:
    #         print('Far')
    for n, x in enumerate(current_coords):
        # print(x, current_depth.shape, sum_list(current_depth, x[0], x[1], x[2], x[3]))
        # print("Object:", sum_list(current_depth, x[0], x[1], x[2], x[3]) /  (x[2] * x[3]))
        # print("Entire Image", sum_list(current_depth, 0, 0, 400, 300) / 120000)
        #     print(current_depth)
        # print("X =", x)
        if sum_list(current_depth, 0, 0, 400, 300) / 120000 < sum_list(current_depth, x[0], x[1], x[2], x[3]) / (x[2] * x[3]):
            print('Near')
            flag = 1
            speak.speak(current_obj[n], current_pos[n])
        else:
            print('Far')
            flag = 0
            speak.speak(current_obj[n], current_pos[n])


if __name__ == '__main__':
    buffer = 0
    obj = []
    coords = []
    depth = []
    pos = []

    cap = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)

    start1 = time.time()
    left = cv2.imread(r'F:\PyCharm Projects\Audio Assistance to the Visually Impaired\Final\Stereo_Vision\images\a\test\left_test0.png')
    right = cv2.imread(r'F:\PyCharm Projects\Audio Assistance to the Visually Impaired\Final\Stereo_Vision\images\a\test\right_test0.png')
    to_object_detection(right)
    to_depth_estimation(left, right)
    print(time.time() - start1)
    to_speech()
    print(time.time() - start1)
    # while True:
    #     ret, left = cap.read()
    #     ret2, right = cap2.read()
    #     cv2.imshow('left', left)
    #     cv2.imshow('right', right)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break
    #     to_object_detection(right)
    #     to_depth_estimation(left, right)
    #     # print(time.time() - start1)
    #     to_speech()
        # print(time.time() - start1)
    #
    #     object_thread = threading.Thread(target=to_object_detection, args=(right, ))
    #     object_thread.start()
    #     depth_thread = threading.Thread(target=to_depth_estimation, args=(left, right))
    #     depth_thread.start()
    #     speech_thread = threading.Thread(target=to_speech())
    #     speech_thread.start()

