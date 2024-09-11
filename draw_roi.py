import cv2
import numpy as np

roi = []
drawing = False

def draw_roi(event, x, y, flags, param):
    global roi, drawing, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image_copy = image.copy()
            cv2.rectangle(image_copy, roi[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Frame", image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi.append((x, y))
        cv2.rectangle(image, roi[0], roi[1], (0, 255, 0), 2)
        cv2.imshow("Frame", image)

def get_roi_coordinates(roi):
    if len(roi) < 2:
        return None
    return (min(roi[0][0], roi[1][0]), min(roi[0][1], roi[1][1]), 
            abs(roi[1][0] - roi[0][0]), abs(roi[1][1] - roi[0][1]))
# Specify the path to your video file
video_path = "cash_no_invoice_30sec.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, image = cap.read()
if not ret:
    print("Failed to read the video")
    exit()

image_copy = image.copy()
roi = []
drawing = False

# Create a window and set the callback function for mouse events
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_roi)

print("Draw the ROI. Press 'q' when done.")

while True:
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()

# Get the ROI coordinates
roi_coords = get_roi_coordinates(roi)

if roi_coords:
    print(f"ROI coordinates (x, y, width, height): {roi_coords}")
    
    # Extract the ROI
    x, y, w, h = roi_coords
    roi_image = image[y:y+h, x:x+w]
    
    # Display the ROI
    cv2.imshow("ROI", roi_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Invalid ROI. Please try again.")

# Release the video capture object
cap.release()