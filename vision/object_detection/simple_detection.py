import cv2
import argparse
import sys
import numpy as np 
import os.path 
import time

# Analyse one image out of 15
SKIP_IMAGE = 10 


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] -1] for i in net.getUnconnectedOutLayers()]

# Load Yolo
net = cv2.dnn.readNet("models/yolov3-tiny.cfg", "models/yolov3-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classes = []
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Process output
outputFile = 'output/results.avi'

cap = cv2.VideoCapture('videos/BBQ1.mp4')
video_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

img_count = 0
skipper = 0
start_analyse_time = time.time()
while cv2.waitKey(1) < 0:
    start_timeframe = time.time()
    # Get frame from the video
    hasFrame, img = cap.read()

    if not hasFrame:
        print('Done processing !!!')
        cv2.waitKey(3000)
        break

    origImg = img
    height, width, channels = img.shape

    if (skipper % SKIP_IMAGE) == 0:
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=True)
        # Set input of the network
        net.setInput(blob)

        # Run in the net
        outs = net.forward(getOutputsNames(net))

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Remove duplicates
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    #combinedImages = np.concatenate((origImg, img), axis=1)
    # Write image
    end_timeframe = time.time()
    print('img #:{0} | FPS : {1:%2f}'.format(img_count, 1/(end_timeframe - start_timeframe)))
    img_count = img_count + 1
    skipper += 1
    video_writer.write(img.astype(np.uint8))
processing_time = time.time() - start_analyse_time
print('analyse video in {0}s | {1:%2f} FPS'.format(processing_time, img_count/processing_time))
