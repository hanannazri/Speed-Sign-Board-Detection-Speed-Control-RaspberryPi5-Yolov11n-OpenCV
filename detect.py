# Importing required Python libraries
import os  # For file path operations
import sys  # For system-related operations like exiting the program
import argparse  # For parsing command-line arguments
import glob  # For handling file path patterns
import time  # For measuring time and calculating FPS
import atexit
import signal
# OpenCV for image and video handling
import cv2
import numpy as np  # For numerical operations
from ultralytics import YOLO  # Importing YOLO model from the Ultralytics package

# Define command-line arguments that users can provide when running the script
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)  # Path to the YOLO model
parser.add_argument('--source', help='Image source', required=True)  # Input source: file, folder, video, or camera
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5, type=float)  # Confidence threshold
parser.add_argument('--resolution', type=str, default='640x480', help='Resolution widthxheight')  # Desired resolution
args = parser.parse_args()
resW, resH = map(int, args.resolution.split('x'))  # Extract width and height from resolution string
parser.add_argument('--record', action='store_true', help='Record results')  # Optional: record video
args = parser.parse_args()

# Extract parsed values into variables
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Output file for storing detected speed signs
SPEED_FILE = "detected_speed.txt"  # File to store detected speed sign info
CONFIDENCE_THRESHOLD = 0.65  # Higher threshold for confirming speed sign detection

def clear_detected_speed_file():
    open(SPEED_FILE, 'w').close()  # Clear the file

# Register cleanup for normal exit
atexit.register(clear_detected_speed_file)

# Register cleanup for Ctrl+C or termination signals
def handle_signal(signum, frame):
    clear_detected_speed_file()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_signal)  # Termination (e.g., system shutdown)

# Check if the provided model path is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(0)  # Exit if the model is not found

# Load the YOLO model
model = YOLO(model_path, task='detect')
labels = model.names  # Get class labels (like Speed20, Speed40, etc.)

# List of supported image and video file extensions
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# Determine the type of source input
if os.path.isdir(img_source):
    source_type = 'folder'  # A folder of images
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'  # Single image file
    elif ext in vid_ext_list:
        source_type = 'video'  # Video file
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)  # Exit if unsupported file extension
elif 'usb' in img_source:
    source_type = 'usb'  # USB camera input like usb0, usb1...
    usb_idx = int(img_source[3:])  # Extract USB camera index
elif 'picamera' in img_source:
    source_type = 'picamera'  # Raspberry Pi camera input like picamera0
    picam_idx = int(img_source[8:])  # Extract camera index
else:
    print(f'Input {img_source} is invalid.')
    sys.exit(0)  # Exit for invalid source type

# Enable resizing if a resolution was provided
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Set up video recorder if recording is enabled
if record:
    if source_type not in ['video', 'usb', 'picamera']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    record_name = 'demo1.avi'  # Output video filename
    record_fps = 30  # Frames per second for recording
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Load image/video/camera source
if source_type == 'image':
    imgs_list = [img_source]  # Just one image
elif source_type == 'folder':
    imgs_list = glob.glob(os.path.join(img_source, '*'))  # Get all files in folder
    imgs_list = [f for f in imgs_list if os.path.splitext(f)[1] in img_ext_list]  # Filter supported images
elif source_type == 'video' or source_type == 'usb':
    cap_arg = img_source if source_type == 'video' else usb_idx  # Set input for video capture
    cap = cv2.VideoCapture(cap_arg)  # OpenCV video capture object
    if user_res:
        cap.set(3, resW)  # Set width
        cap.set(4, resH)  # Set height
elif source_type == 'picamera':
    from picamera2 import Picamera2  # Import Raspberry Pi camera lib
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))  # Set format/resolution
    cap.start()  # Start camera

# Define a set of colors for drawing bounding boxes
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

# Initialize performance tracking variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200  # Average FPS over last 200 frames
img_count = 0  # To keep track of images processed

# Dictionary to count high-threshold detections for each class
class_detection_count = {'Speed20': 0, 'Speed40': 0, 'Speed60': 0, 'Stop': 0}
detected_class = {"Speed20": False, "Speed40": False, "Speed60": False, "Stop": False}

# Begin processing loop
try:
    while True:
        t_start = time.perf_counter()  # Start time for FPS calculation

        # Load next frame depending on the source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images processed. Exiting.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count += 1
        elif source_type == 'video' or source_type == 'usb':
            ret, frame = cap.read()  # Read frame
            if not ret:
                print('End of video or camera disconnected. Exiting.')
                break
        elif source_type == 'picamera':
            frame = cap.capture_array()  # Capture frame
            if frame is None:
                print('Error capturing frame from Picamera. Exiting.')
                break

        # Resize frame if necessary
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Run YOLO model on the frame
        results = model(frame, verbose=False)
        detections = results[0].boxes  # Get detection boxes
        object_count = 0
        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)  # Get box coordinates
            conf = detections[i].conf.item()  # Confidence
            classidx = int(detections[i].cls.item())  # Class ID
            classname = labels[classidx]  # Class name

            if conf > min_thresh:
                color = bbox_colors[classidx % 10]  # Assign color
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)  # Draw box
                label = f'{classname}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(xyxy[1], labelSize[1] + 10)
                cv2.rectangle(frame, (xyxy[0], label_ymin - labelSize[1] - 10),
                              (xyxy[0] + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xyxy[0], label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                object_count += 1

                # Check if confidence is above threshold
                if conf > CONFIDENCE_THRESHOLD:
		    if classname in ['Speed20','Speed40','Speed60']:
		        if class_detection_count[classname] < 3:  # Less than 3 detections for this class
			    class_detection_count[classname] += 1  # Increment the count
                    elif classname =='Stop':
		        if class_detection_count[classname] <5:
		            class_detection_count[classname] += 1   
                    else:
                        # Only log to file and reset the count if it's a new class
                        if detected_class[classname] == False:  # New class detected
                            with open('detected_speed.txt', 'w') as f:
                                f.write(f'{classname}\n')  # Write class name to file
                            detected_class[classname] = True  # Add to the list of detected classes
                            for key in detected_class:
                                if key != classname:
                                    detected_class[key] = False
                        # Reset the count after threshold is reached to avoid repeated detections
                    if (classname in ['Speed20', 'Speed40', 'Speed60'] and class_detection_count[classname] == 3) or (classname == 'Stop' and class_detection_count[classname] == 5):
                            class_detection_count[classname] = 0

        # Display FPS and object count on frame
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show the frame with detections
        cv2.imshow('YOLO detection results', frame)

        # Save the frame if recording is enabled
        if record and frame is not None:
            recorder.write(frame)

        # Check for quit key
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate_calc = 1.0 / (t_stop - t_start)
        frame_rate_buffer.append(frame_rate_calc)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = np.mean(frame_rate_buffer)  # Average FPS over buffer

# Final cleanup when the script is exiting
finally:
    if source_type == 'video' or source_type == 'usb':
        if 'cap' in locals() and cap.isOpened():
            cap.release()
    elif source_type == 'picamera':
        if 'cap' in locals():
            cap.stop()
    if 'recorder' in locals() and record:
        recorder.release()
    cv2.destroyAllWindows()  # Close OpenCV display window
	
