import time
from gpiozero import Robot, Motor, PWMOutputDevice

# Motor setup - using BCM GPIO numbers with physical pin numbers in comments
left_motor_forward = 16  # PIN 36
left_motor_backward = 12 # PIN 32
left_motor_enable_pin = 18  # PIN 12 (ENA) - Store the PIN NUMBER

right_motor_forward = 6   # PIN 31
right_motor_backward = 5  # PIN 29
right_motor_enable_pin = 21  # PIN 40 (ENB) - Store the PIN NUMBER

# Initialize PWMOutputDevice for speed control
right_motor_pwm = PWMOutputDevice(right_motor_enable_pin)
left_motor_pwm = PWMOutputDevice(left_motor_enable_pin)

# Create the Motor objects, WITHOUT the enable parameter
right = Motor(forward=right_motor_forward, backward=right_motor_backward)
left = Motor(forward=left_motor_forward, backward=left_motor_backward)

# Create the Robot object
robot = Robot(right=right, left=left)

# Initialize robot speed (duty cycle)
DEFAULT_SPEED = 0.7

# File to read the detected speed from
SPEED_FILE = "detected_speed.txt"

def set_robot_speed(speed):
    """
    Update the robot's speed (duty cycle from 0.0 to 1.0).
    """
    right_motor_pwm.value = speed
    left_motor_pwm.value = speed

def process_detected_speed(detected_speed):
    if detected_speed == "Speed60":
        print("Speed 60 received: Moving forward at high speed.")
        set_robot_speed(0.9)
        robot.forward()
    elif detected_speed == "Speed40":
        print("Speed 40 received: Moving forward at moderate speed.")
        set_robot_speed(0.5)
        robot.forward()
    elif detected_speed == "Speed20":
        print("Speed 20 received: Moving forward at low speed.")
        set_robot_speed(0.3)
        robot.forward()
    elif detected_speed == "Stop":
        print("STOP sign received: Stopping robot.")
        set_robot_speed(0.0)
        robot.stop()
    elif detected_speed == "None":
        print("No speed sign detected: Maintaining default forward speed.")
        set_robot_speed(DEFAULT_SPEED)
        robot.forward(DEFAULT_SPEED)
    else:
        print(f"Received unknown speed: {detected_speed}")
        set_robot_speed(DEFAULT_SPEED)
        robot.forward(DEFAULT_SPEED)

if __name__ == "__main__":
    try:
        # Initialize default speed
        set_robot_speed(DEFAULT_SPEED)
        robot.forward(DEFAULT_SPEED)

        while True:
            try:
                # Read the detected speed from the file
                with open(SPEED_FILE, "r") as f:
                    detected_speed_str = f.readline().strip()
                    process_detected_speed(detected_speed_str)
            except FileNotFoundError:
                print(f"Error: {SPEED_FILE} not found. Maintaining default speed.")
                process_detected_speed("None")
            except Exception as e:
                print(f"Error reading speed from file: {e}")
                process_detected_speed("None")

            time.sleep(0.1) # Small delay to avoid excessive file reading

    finally:
        robot.stop()
        right_motor_pwm.close()
        left_motor_pwm.close()
