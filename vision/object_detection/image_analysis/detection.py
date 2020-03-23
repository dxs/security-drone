import cv2
import time
import numpy as np

class Detection:

    def __init__(self):
        self.TINY_MODEL = False
        self.initialize_model()

    def initialize_model(self):
        # Load Yolo
        if self.TINY_MODEL:
            self.cfg = "models/yolov3-tiny.cfg"
            self.weights = "models/yolov3-tiny.weights"
        else:
            self.cfg = "models/yolov3.cfg"
            self.weights = "models/yolov3.weights"


        self.net = cv2.dnn.readNet(self.cfg, self.weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.classes = []
        with open("models/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()

        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] -1] for i in self.net.getUnconnectedOutLayers()]

    def process_image(self, img):
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=True)

        self.net.setInput(blob)

        outs = self.net.forward(self.getOutputsNames())

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.01:
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
                    print("Class id : {}, corresponds to : {} (condidence : {:.02f})".format(class_id, str(self.classes[class_id]), float(confidence)))


        # Remove duplicates

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        return img
