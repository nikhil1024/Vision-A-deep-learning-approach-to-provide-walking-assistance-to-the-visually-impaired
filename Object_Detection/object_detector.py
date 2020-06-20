import cv2
import numpy as np
import time


t = 0


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    global t
    label = str(classes[class_id]) + ' ' + str(format(confidence, '.2f'))
    if t == 0:
        color = [255, 0, 0]
    elif t == 1:
        color = [0, 255, 0]
    elif t == 2:
        color = [0, 0, 255]
    else:
        color = [0, 255, 0]
    # print(color)
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    t += 1


def detect_objects(image):
    global classes
    objects, positions, coords = [], [], []
    image = cv2.resize(image, (400, 300))
    # image = cv2.resize(cv2.imread(r'dog.jpg'), (400, 300))

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    with open('./Object_Detection/yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    # print(COLORS)
    net = cv2.dnn.readNet('./Object_Detection/yolov3.cfg', './Object_Detection/yolov3.weights')
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    start = time.time()

    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    # print(outs)
    # print(outs[2].shape, len(outs))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    loop = 0
    for out in outs:
        loop += 1
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # print('\n\n\n', loop, detection)
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    total_time = time.time() - start
    # print(total_time)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        # print(classes[class_ids[i]] + ' detected with an accuracy of ' + str(confidences[i] * 100) + ' %')
        objects.append(classes[class_ids[i]])

        center_x = x + (w / 2)
        # center_y = (y + h) / 2
        # print("co ", x, y, w, h, image.shape)
        coords.append([x, y, w, h])
        if center_x < 133:
            # print("right")
            # print(center_x)
            positions.append('left')
        elif center_x > 266:

            positions.append('right')
            # print(center_x)
            # print("left")
        else:
            positions.append('front')

    # cv2.imwrite("object-detection.jpg", image)
    cv2.imshow('a', image)
    cv2.waitKey(0)


    cv2.destroyAllWindows()
    return objects, coords, positions


classes = None
objects, positions, coords = [], [], []
# detect_objects()
