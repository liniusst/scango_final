import cv2
from glob import glob
from ultralytics import YOLO
import utils
import torch
import numpy as np

# pick pre-trained model
np_model = YOLO("best.pt")

results = {}

# read video by index
videos = glob("inputs/*.mp4")
video = cv2.VideoCapture(videos[0])

ret = True
frame_number = -1
vehicles = [2, 3, 5, 7]
license_plate_detected = False

allow_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def process_detection(image, box):
    x1, y1, x2, y2, _, _ = box
    region_of_interest = image[int(y1) : int(y2) + 5, int(x1) : int(x2) + 5]
    gray_image = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray_image, 11, 17, 17)
    _, thresholded = cv2.threshold(bfilter, 64, 255, cv2.THRESH_BINARY_INV)
    return thresholded


# read the 10 first frames
while ret:
    frame_number += 1
    ret, frame = video.read()

    if ret and frame_number < 50:
        # results[frame_number] = {}

        # license plate detector for the whole frame
        with torch.no_grad():
            license_plate = np_model(frame)[0]

        img_tresh = [
            process_detection(frame, box) for box in license_plate.boxes.data.tolist()
        ]
        for img in img_tresh:
            cnts, _ = cv2.findContours(
                np.vstack(img).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)

            results = utils.reader.readtext(
                np.vstack(img), allowlist=allow_list, paragraph=False
            )

        # process license plates
        # for license_plate in license_plates.boxes.data.tolist():
        #     (
        #         plate_x1,
        #         plate_y1,
        #         plate_x2,
        #         plate_y2,
        #         plate_score,
        #         _,
        #     ) = license_plate

        # crop plate from the frame
        # plate = frame[int(plate_y1) : int(plate_y2), int(plate_x1) : int(plate_x2)]

        # de-colorize
        # plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # posterize
        # _, plate_threshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # OCR
        # np_text, np_score = utils.read_license_plate(plate_threshold)

        # if plate could be read, write results
        # if np_text is not None:
        #     results[frame_number][1] = {
        #         "license_plate": {
        #             "bbox": [plate_x1, plate_y1, plate_x2, plate_y2],
        #             "bbox_score": plate_score,
        #             "number": np_text,
        #             "text_score": np_score,
        #         },
        #     }

# write results to CSV
# utils.write_csv(results, "./results.csv")
print(results)
video.release()
