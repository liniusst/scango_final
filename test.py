from paddleocr import PaddleOCR, draw_ocr, download_with_progressbar
import torch
import os
from ultralytics import YOLO
import cv2
import numpy as np

torch.device("cpu")
model_path = os.path.join(".", "weights", "best.pt")
model = YOLO(model_path)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

np.int = np.int_

img_path = "1.jpeg"
image = cv2.VideoCapture(img_path)
ret, img = image.read()

with torch.no_grad():
    detections = model(img)[0]

for box in detections.boxes.data.tolist():
    x1, y1, x2, y2, track_id, score = box
    region_of_interest = img[int(y1) : int(y2) + 5, int(x1) : int(x2)]
    gray_frame = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_frame, 64, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    morphology = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

result = ocr.ocr(morphology, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)
print(result)


# result = ocr.ocr(img_path)
