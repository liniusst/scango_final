import cv2
import easyocr

# Create an EasyOCR reader
reader = easyocr.Reader(["en"], gpu=False)  # Use 'en' for English language

# Create a VideoCapture object
vidcap = cv2.VideoCapture(0)

# Check if the connection with the camera is successful
if vidcap.isOpened():
    while True:
        ret, frame = vidcap.read()  # Capture a frame from live video

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Perform any additional preprocessing steps as needed (e.g., resizing, blurring)

        # Use EasyOCR to detect text (license plate) in the frame
        results = reader.readtext(frame)

        # Iterate through the results and print the detected text
        for bbox, text, prob in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Draw the bounding box around the detected text
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Print the detected text
            print("Detected License Plate:", text)

        # Display the frame
        cv2.imshow("License Plate Detection", frame)

        # Press 'q' to break out of the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close windows
    vidcap.release()
    cv2.destroyAllWindows()

# Print an error if the connection with the camera is unsuccessful
else:
    print("Cannot open camera")
