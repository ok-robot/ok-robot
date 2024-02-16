import argparse
import sys
import os

from manipulation import ObjectHandler
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='./checkpoints/checkpoint_detection.tar', help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.07, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.05, help='Gripper height')
parser.add_argument('--port', type=int, default = 5556, help='port')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
parser.add_argument('--open_communication', action='store_true', help='Use image transferred from the robot')
parser.add_argument('--max_depth', type=float, default = 2, help='Maximum depth of point cloud')
parser.add_argument('--min_depth', type=float, default = 0, help='Maximum depth of point cloud')
parser.add_argument('--sampling_rate', type=float, default = 1, help='Sampling rate of points [<= 1]')
parser.add_argument('--environment', default = './example_data', help='Environment name')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.2, cfgs.max_gripper_width))

def check_license_folder():
    license_path = "./license"
    if (not os.path.exists(license_path)) or (len(os.listdir(license_path)) != 4):
        print("Couldn't find the license folder in the /src directory.")
        print("Group the license related .json, .lic, .public_key, .signature files into a license folder and place it inside the /src directory")
        sys.exit(1)

def demo():
    # Checking the proper license folder placement.
    check_license_folder()

    object_handler = ObjectHandler(cfgs)
    while True:
        object_handler.manipulate()

if __name__ == "__main__":
    demo()
