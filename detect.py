import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5, type=float)
parser.add_argument('--resolution', type=str, default='640x480', help='Resolution widthxheight')
args = parser.parse_args()
resW, resH = map(int, args.resolution.split('x'))
parser.add_argument('--record', action='store_true', help='Record results')
args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# File to store the detected speed
SPEED_FILE = "detected_speed.txt"
CONFIDENCE_THRESHOLD = 0.65  # Increased confidence threshold for speed detection

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(0)

# Load the model
model = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid.')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Set up recording if enabled
if record:
    if source_type not in ['video', 'usb', 'picamera']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = glob.glob(os.path.join(img_source, '*'))
    imgs_list = [f for f in imgs_list if os.path.splitext(f)[1] in img_ext_list]
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video':
        cap_arg = img_source
    elif source_type == 'usb':
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
try:
    while True:
        t_start = time.perf_counter()

        # Load frame from image source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images processed. Exiting.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count += 1
        elif source_type == 'video' or source_type == 'usb':
            ret, frame = cap.read()
            if not ret:
                print('End of video or camera disconnected. Exiting.')
                break
        elif source_type == 'picamera':
            frame = cap.capture_array()
            if frame is None:
                print('Error capturing frame from Picamera. Exiting.')
                break

        # Resize frame
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        high_confidence_detections = {}

        # Process detections for speed signs
        for detection in detections:
            class_idx = int(detection.cls.item())
            class_name = labels[class_idx]
            conf = detection.conf.item()

            if conf > CONFIDENCE_THRESHOLD and class_name in ["Speed60", "Speed40", "Speed20", "Stop"]:
                print(f"High Confidence Speed Sign Detected: {class_name} with Confidence: {conf:.2f}")
                high_confidence_detections[class_name] = high_confidence_detections.get(class_name, 0) + 1

        most_frequent_speed = None
        max_count = 0
        for speed, count in high_confidence_detections.items():
            if count > max_count:
                most_frequent_speed = speed
                max_count = count

        # Write the most frequently detected high-confidence speed to the file
        with open(SPEED_FILE, "w") as f:
            if most_frequent_speed:
                f.write(most_frequent_speed)
            else:
                f.write("None")

        # Draw bounding boxes and labels for all detected objects
        object_count = 0
        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            conf = detections[i].conf.item()
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            if conf > min_thresh:
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                label = f'{classname}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(xyxy[1], labelSize[1] + 10)
                cv2.rectangle(frame, (xyxy[0], label_ymin - labelSize[1] - 10),
                              (xyxy[0] + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xyxy[0], label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                object_count += 1

        # Calculate and draw FPS
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('YOLO detection results', frame)
        if record and frame is not None:
            recorder.write(frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate_calc = 1.0 / (t_stop - t_start)
        frame_rate_buffer.append(frame_rate_calc)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = np.mean(frame_rate_buffer)

finally:
    if source_type == 'video' or source_type == 'usb':
        if 'cap' in locals() and cap.isOpened():
            cap.release()
    elif source_type == 'picamera':
        if 'cap' in locals():
            cap.stop()
    if 'recorder' in locals() and record:
        recorder.release()
    cv2.destroyAllWindows()
