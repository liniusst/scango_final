from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
import time
from sort.sort import Sort
import threading

torch.device("cpu")
car_model_path = os.path.join(".", "weights", "yolov8n.pt")
car_model = YOLO(car_model_path)

model_path = os.path.join(".", "weights", "best.pt")
model = YOLO(model_path)

np.int = np.int_


class detect_license_plate:
    def __init__(self, conf_score: float, max_frames: int) -> None:
        self.conf_score = conf_score
        self.motion_tracker = Sort()
        self.vehicle_ids = [2, 3, 5, 7]
        self.max_frames = max_frames
        self.ocr = None
        self.init_thread = threading.Thread(target=self._initialize_ocr)
        self.init_thread.start()

    def _process_thresholded(self, region_of_interest):
        gray_frame = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_frame, 64, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        tresh_morphology = cv2.morphologyEx(
            thresholded, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        return tresh_morphology

    def _initialize_ocr(self):
        if self.ocr is None:
            self.ocr = PaddleOCR(
                use_angle_cls=True, lang="en", show_log=False, cls_thresh=0.8
            )

    def _read_license_plate(self, img_tresh):
        if self.init_thread.is_alive():
            self.init_thread.join()
        result = self.ocr.ocr(img_tresh, cls=True)

        license_plates = ""
        confidence_scores = []

        result = sorted(result, key=lambda x: x[0][0][0])

        for detection in result:
            bbox = detection[0]
            text, score = detection[1]

            if score >= self.conf_score:
                license_plates += text
                confidence_scores.append(score)

        license_plates = license_plates.upper().replace(" ", "")

        avg_confidence = (
            np.round(np.mean(confidence_scores), 2) if confidence_scores else 0.0
        )

        return license_plates, avg_confidence

    def _map_car(self, plate, tracking_ids):
        x1, y1, x2, y2, score, class_id = plate

        for j in range(len(tracking_ids)):
            x1car, y1car, x2car, y2car, car_id = tracking_ids[j]

            if x1 > x1car and y1 > y1car and x2 < x2car and y2 < y2car:
                car_index = j
                return tracking_ids[car_index]

        return -1, -1, -1, -1, -1

    def _check_status(self, result) -> bool:
        if result is not None:
            license_number, license_number_score = result
        else:
            license_number, license_number_score = "-1", "-1"

        if license_number != "-1" and float(license_number_score) >= self.conf_score:
            status = True
        else:
            status = False

        return status, license_number, license_number_score

    def process_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        results = {}

        frame_nmb = 0
        ret = True

        while ret and frame_nmb < self.max_frames:
            ret, frame = video.read()
            if not ret:
                break

            results[frame_nmb] = {}

            start_time = time.time()

            with torch.no_grad():
                vehicles = car_model(frame)[0]

            vehicles_list = []
            for vehicle in vehicles.boxes.data.tolist():
                x1, y1, x2, y2, vehicle_bbox_score, class_id = vehicle
                if int(class_id) in self.vehicle_ids:
                    vehicles_list.append([x1, y1, x2, y2, vehicle_bbox_score])
                    vehicle_bbox_score = np.round(np.mean(vehicle_bbox_score), 2)

            tracking_ids = self.motion_tracker.update(np.asarray(vehicles_list))

            with torch.no_grad():
                plates = model(frame)[0]

            for plate in plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = plate

                x1car, y1car, x2car, y2car, car_id = self._map_car(plate, tracking_ids)
                if car_id != -1:
                    region_of_interest = frame[int(y1) : int(y2), int(x1) : int(x2)]

                    tresh_img = self._process_thresholded(region_of_interest)
                    result = self._read_license_plate(tresh_img)

                    status = self._check_status(result)
                    bool_status, license_number, license_number_score = status

                    end_time = time.time()
                    detection_time = round((end_time - start_time) * 1000, 2)

                    if bool_status:
                        results[frame_nmb][car_id] = {
                            "plate": {
                                "license_plate": license_number,
                                "vehicle_conf_score": vehicle_bbox_score,
                                "plate_conf_score": license_number_score,
                                "detection_time": detection_time,
                            },
                        }

                    print(results)
            frame_nmb += 1
        video.release()


if __name__ == "__main__":
    video_path = "detection/video.mp4"
    license_plate_detector = detect_license_plate(conf_score=0.91, max_frames=4)
    license_plate_detector.process_video(video_path)
