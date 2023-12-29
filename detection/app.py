from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
import csv
from paddleocr import PaddleOCR, draw_ocr, download_with_progressbar
import time


torch.device("cpu")
model_path = os.path.join(".", "weights", "best.pt")
model = YOLO(model_path)
np.int = np.int_


class detect_license_plate:
    def __init__(self, conf_score) -> None:
        self.allow_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.conf_score = conf_score
        self.ocr = PaddleOCR(
            use_angle_cls=True, lang="en", show_log=False, cls_thresh=0.8
        )

    def _process_detection(self, frame, box):
        try:
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
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def _write_csv(self, results, output_path):
        try:
            file_exists = os.path.exists(output_path)

            with open(output_path, "a", newline="") as f:
                fieldnames = ["plate_number", "confidence_level"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                for result in results:
                    writer.writerow(result)

        except Exception as e:
            print(f"An error occurred while writing to CSV: {e}")

    def _read_license_plate(self, img_tresh):
        try:
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

        except Exception as e:
            print(f"An error occurred while reading license plate: {e}")
            return None

    def process_video(self, video_path):
        try:
            video = cv2.VideoCapture(video_path)

            ret = True
            frame_number = -1

            fps = video.get(cv2.CAP_PROP_FPS)
            desired_fps = 3
            frame_interval = int(round(fps / desired_fps))

            all_results = []

            start_time = time.time()

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

            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Processing time: {processing_time:.2f} seconds")

        except Exception as e:
            print(f"An error occurred during video processing: {e}")
            return None

        finally:
            video.release()

        return all_results


if __name__ == "__main__":
    video_path = "images/2.jpeg"
    license_plate_detector = detect_license_plate(conf_score=0.01)
    license_plate_detector.process_video(video_path)
