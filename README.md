# scango_final
importanize

**scango_final** is a Python project that combines YOLO (You Only Look Once) for object detection, the SORT (Simple Online and Realtime Tracking) algorithm for tracking, and PaddleOCR for license plate recognition. This project aims to detect and track vehicles, recognize license plates, and provide associated information.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)

## Overview

The project uses YOLO for vehicle detection, SORT for tracking, and PaddleOCR for license plate recognition. The main components include:

- **Object Detection:** Utilizes YOLO to detect vehicles in video frames.
- **Tracking:** Employs SORT for tracking detected vehicles over consecutive frames.
- **License Plate Recognition:** Uses PaddleOCR to recognize license plates within the tracked vehicles' regions.

## Dependencies

Ensure you have the following dependencies installed:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics): YOLO implementation for object detection.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): OCR library for license plate recognition.
- [SORT](https://github.com/abewley/sort): SORT algorithm for object tracking.
- OpenCV
- NumPy
- Torch


### Object Tracking

- **Input**: 448x640, 1 car, 1 train
- **Processing Time**: 259.6ms (14.4ms preprocess, 259.6ms inference, 25.7ms postprocess per image at shape (1, 3, 448, 640))

### License Plate Detection

- **Input**: 448x640, 1 license plate
- **Processing Time**: 174.5ms (4.4ms preprocess, 174.5ms inference, 2.0ms postprocess per image at shape (1, 3, 448, 640))

#### Detected License Plate Information

```python
{
    0: {
        1.0: {
            'plate': {
                'license_plate': 'B2228HM',
                'vehicle_conf_score': 0.35,
                'plate_conf_score': 0.99,
                'detection_time': 504.93
            }
        }
    }
}
