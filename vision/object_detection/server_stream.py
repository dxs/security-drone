import imagezmq
import cv2
import argparse
import sys
import numpy as np 
import os.path 
import time
from image_analysis.detection import Detection

smart = Detection()

image_hub = imagezmq.ImageHub()
while True:
    rpi_name, image = image_hub.recv_image()
    print("received_image")
    labelled_image = smart.process_image(image)
    print("processed_image")
    cv2.imshow(rpi_name, labelled_image)
    print("Showed image")
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')  # this statement is missing from your while True loo
