# Speed Sign Board Detection using YOLO11n and Speed Control System

This project focuses on speed control in real-life scenarios like hospital zones, school zones, and highways, where maintaining appropriate speed limits is crucial for safety. The goal is to build a system that can detect speed signboards and automatically adjust a car’s speed—just like how smart vehicles behave in the real world.

To achieve this, our team developed a custom YOLOv11n model, a lightweight deep learning-based object detection algorithm. It is trained to recognize four types of speed signboards: 20, 40, 60, and STOP.

In the prototype setup, the trained model runs in a simulated environment representing real-world locations. When it detects a speed signboard, the system automatically sends signals to a motor, adjusting the speed or movement of a small car—either slowing it down, speeding it up, or stopping it based on the sign detected.

It's a great example of how computer vision and robotics can come together to create smart, responsive systems.

## Dataset Preparation

The dataset used to train the model was created entirely from scratch. I manually captured images of speed signboards (20, 40, 60, STOP) in various settings. These images were then annotated and augmented using Roboflow, resulting in a clean and diverse dataset.

- *Total Images*: 10,399
- *Classes*: 4 (20, 40, 60, STOP)
- *Per Class Samples*: ~2,600 images
- *Preprocessing*: Roboflow for annotation & augmentation

## Model Training

The base model used in this project is YOLOv11n, pre-trained on the COCO dataset. Fine-tuning was carried out in Google Colab using the Ultralytics YOLOv11n implementation. The model was trained for 100 epochs with an image size of 640 and a patience value of 100 to ensure optimal performance. After training and optimization using PyTorch, the final model size was reduced to just 5.2 MB, making it lightweight and well-suited for deployment on resource-constrained devices like the Raspberry Pi 5.

**Performance Metrics:**
- *Precision*: 96.9%
- *Recall*: 93.0%
- *mAP@0.5*: 95.7%
- *mAP@0.5:0.95*: 85.2%

## Components Setup

- *Raspberry Pi 5* - The Main Processing Unit
- *Camera Module*  - To Capture real-time video for signboard detection.
- *LN298N Motor Driver and 4BO Motors* - To control the car's movement
- *Battery Powered Power Supply* - To Power the Raspberry Pi and Motors

## Installation Steps

- Connect the GPIO Pins to the Motor Driver in such a way that:
  |**GPIO Pins**|**Motor Driver**|
  |---------|------------|
  | GPIO12 (PIN 12)| ENA | 
  | GPIO21 (PIN 40)| ENB |
  | GPIO16 (PIN 36)| IN1 |
  | GPIO12 (PIN 32)| IN2 |
  | GPIO 6 (PIN 31)| IN3 |
  | GPIO 5 (PIN 29)| IN4 |
  | GND    (PIN39) | GND |

## Deployment

- *OpenCV* - used for video processing tasks, for reading from the live camera, pre-processing frames for the YOLO Model, and potentially draw bounding boxes around the detected signs.
- *gpiozero* - used for interfacing with Raspberry Pi's GPIO Pin, which is used to control the car's motors.    

## Usage 

Directions to run the code
1. Unzip the whole repository and make it your current directory 
2. Install all the required dependencies using the requirments.txt file
    * Open the Terminal in Raspberry Pi OS
    * **Type** - `pip install -r requirements.txt`
3. For running on USB Camera 
    * **Type** - `python detect.py Models/speed_sign_board.pt --source usb0`
    * Or if you're using Picamera
    * **Type** - `python detect.py --model Models/speed_sign_board.pt --source picamera0 --resolution 640x840`
4. For Motor Control
   * Open a new window in terminal
   * **Type** - `python control.py`
  

