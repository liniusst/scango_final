from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
import csv
from paddleocr import PaddleOCR, draw_ocr, download_with_progressbar

torch.device("cpu")
model_path = os.path.join(".", "weights", "best.pt")
model = YOLO(model_path)
np.int = np.int_


class detect_license_plate:
    def __init__(self, conf_score) -> None:
        self.allow_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.conf_score = conf_score
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def _process_detection(self, frame, box):
        x1, y1, x2, y2, track_id, score = box
        region_of_interest = frame[int(y1) : int(y2) + 5, int(x1) : int(x2)]
        gray_frame = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_frame, 64, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        morphology = cv2.morphologyEx(
            thresholded,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        return morphology

    def _write_csv(self, results, output_path):
        file_exists = os.path.exists(output_path)

        with open(output_path, "a", newline="") as f:
            fieldnames = ["plate_number", "confidence_level"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for result in results:
                writer.writerow(result)

    def _read_license_plate(self, img_tresh):
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

            result = {
                "plate_number": license_plates,
                "confidence_level": avg_confidence,
            }

        return result

    def process_video(self, video_path):
        video = cv2.VideoCapture(video_path)

        ret = True
        frame_number = -1

        fps = video.get(cv2.CAP_PROP_FPS)
        desired_fps = 3
        frame_interval = int(round(fps / desired_fps))

        all_results = []

        while ret:
            frame_number += 1
            ret, frame = video.read()
            if not ret:
                break

            if frame_number % frame_interval != 0:
                continue

            with torch.no_grad():
                detections = model(frame)[0]

            for box in detections.boxes.data.tolist():
                img_tresh = self._process_detection(frame, box)
                results = self._read_license_plate(img_tresh)
                confidence_level = results.get("confidence_level")

                if confidence_level >= self.conf_score:
                    all_results.append(results)
                    self._write_csv(all_results, "./results.csv")
                else:
                    break
        video.release()
        return all_results


if __name__ == "__main__":
    video_path = "images/5.jpg"
    license_plate_detector = detect_license_plate(conf_score=0.01)
    license_plate_detector.process_video(video_path)
