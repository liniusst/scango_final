from ultralytics import YOLO
import os
import cv2
import easyocr
import numpy as np
import torch
import csv

torch.device("cpu")
model_path = os.path.join(".", "weights", "best.pt")
model = YOLO(model_path)


class detect_license_plate:
    def __init__(self, conf_score) -> None:
        self.allow_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.conf_score = conf_score

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
        output = self.reader.readtext(
            np.vstack(img_tresh), allowlist=self.allow_list, paragraph=False
        )

        license_plates = ""

        for detection in output:
            bbox, text, score = detection

            if score >= self.conf_score:
                license_plates += text

            text = text.upper().replace(" ", "")
            score = np.round(np.mean(score), 2)

            result = {
                "plate_number": license_plates,
                "confidence_level": score,
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

                cnts, _ = cv2.findContours(
                    np.vstack(img_tresh).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)
                else:
                    break

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
