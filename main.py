import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time
import math

from detect_road import get_lines
import matplotlib.pyplot as plt

gamma = 10e-7
fps = 25
dis_w = 1.5
dis_b = 3.5
dis = 5

class vehicle:
    def __init__(self, type, x, y, frame):
        self.type = type  # type of vehicle
        self.speed = None
        # latest coordinates of vehicle
        self.x = x
        self.y = y

        self.start_frame = -1
        self.end_frame = -1
        self.last_frame = frame

        self.stage = 0


inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)

classes = {'1': 'bicycle', '2': 'car', '3': 'bike', '5': 'bus', '7': 'truck'}
length = {}
list_of_classes = [1, 2, 3, 5, 7]

line1, line2 = get_lines()

x_start = line2[0][1]
x1_end = line2[0][3]
x2_end = line1[0][1]
x3_end = line1[0][3]
print(line1, line2)

with tf.Session() as sess:
    sess.run(model.pretrained())

    cap = cv2.VideoCapture("video.mp4")

    # change the path to your directory or to '0' for webcam
    vehicles = []
    num = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        img = cv2.resize(frame, (416, 416))
        imge = np.array(img).reshape(-1, 416, 416, 3)
        start_time = time.time()
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})

        print("--- %s seconds ---" % (time.time() - start_time))  # to time it
        boxes = model.get_boxes(preds, imge.shape[1:3])

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 700, 700)
        boxes1 = np.array(boxes)

        for j in list_of_classes:  # iterate over classes
            count = 0
            if str(j) in classes:
                lab = classes[str(j)]
            if len(boxes1) != 0:

                # iterate over detected vehicles
                for i in range(len(boxes1[j])):

                    vel = None
                    box = boxes1[j][i]
                    # setting confidence threshold as 40%
                    if boxes1[j][i][4] >= .40:

                        # print(boxes1[j][i])
                        x2 = boxes1[j][i][0]
                        y2 = boxes1[j][i][1]
                        min = 2200
                        temp = None
                        for veh in vehicles:
                            x1 = veh.x
                            y1 = veh.y


                            print(veh.type)
                            # print(x2, y2)

                            if veh.type == lab and (x1<x2 and y1>y2):
                                temp0 = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
                                if temp0 < min:
                                    check = True
                                    temp = veh

                        if temp:
                            print('Updating!!!')
                            min = temp0
                            temp.x = x2
                            temp.y = y2
                            temp.last_frame = num
                            if x2 < x_start and temp.stage == 0:
                                temp.stage = 1
                                temp.start_frame = num
                            if x2 < x1_end and temp.stage == 1:
                                temp.stage = 2
                                temp.end_frame = num
                                print(num, temp.start_frame)
                                temp.speed = 1.5 * 25 * 18 / ((temp.end_frame - temp.start_frame) * 5 + gamma)
                                print(veh.speed)
                            if x2 < x2_end and temp.stage == 2:
                                temp.stage = 3
                                temp.end_frame = num
                                temp.speed = 9 * 25 * 18 / ((temp.end_frame - temp.start_frame) * 5 + gamma)
                            if x2 < x3_end and temp.stage == 3:
                                temp.stage = 4
                                temp.end_frame = num
                                temp.speed = 10.5 * 25 * 18 / ((temp.end_frame - temp.start_frame) * 5 + gamma)
                            vel = temp.speed
                        else :
                            veh = vehicle(lab, boxes[j][i][0], boxes[j][i][1], num)
                            vehicles.append(veh)


                        count += 1
                        cv2.line(img, (0, x_start), (416, x_start), color = (255, 255, 255), thickness=1)
                        cv2.line(img, (0, x1_end), (416, x1_end), color = (255, 255, 255), thickness=1)
                        cv2.line(img, (0, x2_end), (416, x2_end), color = (255, 255, 255), thickness=1)

                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                        cv2.putText(img, lab + 'speed :' + str(vel), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                    lineType=cv2.LINE_AA)

            print(lab, ": ", count)
            for veh in vehicles:
                if num - veh.last_frame > 10:                       # keep vehicle in memory for 25 iterations only
                    vehicles.remove(veh)

        print(len(vehicles))



        # Display the output
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        num += 1

