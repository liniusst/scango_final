import cv2
import numpy as np
import easyocr
import re, os
import torch
from ultralytics import YOLO

torch.device("cpu")
model_path = os.path.join("best.pt")
model = YOLO(model_path)

class detect_license_plate:
    def __init__(self, video_path=0) -> None:
        self.video_path = video_path
        self.result = None

    def _read_frame(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print("Error reading frame.")
                return None
            return frame
        except Exception as e:
            print(f"Error during frame reading: {e}")
            return None

    def _frame_to_thresh(self, frame):
        if frame is None:
            self.frame_thresh = []
            return

        try:
            detections = model(frame)[0]
            self.frame_thresh = [
                cv2.threshold(
                    cv2.cvtColor(
                        frame[int(y1) : int(y2), int(x1) : int(x2)], cv2.COLOR_BGR2GRAY
                    ),
                    64,
                    255,
                    cv2.THRESH_BINARY_INV,
                )[1]
                for x1, y1, x2, y2, _, _ in detections.boxes.data.tolist()
            ]
        except Exception as e:
            print(f"Error during frame processing: {e}")
            self.frame_thresh = []

    def frame_cnts(self):
        if not hasattr(self, "frame_thresh") or not self.frame_thresh:
            return []

        try:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)

            common_width = 100

            resized_frames = [
                cv2.resize(frame, (common_width, frame.shape[0]))
                for frame in self.frame_thresh
            ]

            cnts, _ = cv2.findContours(
                np.vstack(resized_frames).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not cnts:
                return []

            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            output = reader.readtext(np.vstack(self.frame_thresh), paragraph=False)
        except Exception as e:
            print(f"Error during contour extraction or OCR: {e}")
            output = []

        return output

    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            frame_counter = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1

                # Check every 5 frames
                if frame_counter % 2 == 0:
                    self._frame_to_thresh(frame)
                    output = self.frame_cnts()
                    license_plate_list = [
                        re.sub(r"[^A-Z0-9]", "", text)
                        for _, text, text_score in output
                        if text_score > 0.15
                    ]

                    license_plate = "".join(license_plate_list)
                    self.result = {"plate": license_plate}
                    print(self.result)  # You can replace this with your desired processing

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error during video processing: {e}")


# Usage for video from file
try:
    video_path = "apkarpytas.mp4"
    detection = detect_license_plate(video_path)
    detection.process_video()
except Exception as e:
    print(f"Error during license plate detection: {e}")
    # Handle the error or exit the program
