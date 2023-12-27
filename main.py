import ast
import cv2 as cv
import easyocr
from glob import glob
import numpy as np
import pandas as pd
import string
from ultralytics import YOLO

# regular pre-trained yolov8 model for car recognition
# coco_model = YOLO('yolov8n.pt')
coco_model = YOLO("yolov8s.pt")
# yolov8 model trained to detect number plates
np_model = YOLO("best.pt")

# read in test video paths
videos = glob("inputs/*.mp4")

# read video by index
video = cv.VideoCapture(videos[0])

ret = True
frame_number = -1
# all vehicle class IDs from the COCO dataset (car, motorbike, truck) https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
vehicles = [2, 3, 5]
vehicle_bounding_boxes = []

# read the 10 first frames
while ret:
    frame_number += 1
    ret, frame = video.read()

    if ret and frame_number < 10:
        # use track() to identify instances and track them frame by frame
        detections = coco_model.track(frame, persist=True)[0]
        # save cropped detections
        # detections.save_crop('outputs')
        # print nodel predictions for debugging
        # print(results)

        for detection in detections.boxes.data.tolist():
            # print detection bounding boxes for debugging
            # print(detection)
            x1, y1, x2, y2, track_id, score, class_id = detection
            # I am only interested in class IDs that belong to vehicles
            if int(class_id) in vehicles and score > 0.5:
                vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])

# print found bounding boxes for debugging
print(vehicle_bounding_boxes)
video.release()
