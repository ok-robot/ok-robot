import zmq
import numpy as np
from PIL import Image as PILImage

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray,MultiArrayDimension

from utils.communication_utils import send_array, recv_array

class ImagePublisher():

    def __init__(self, camera, socket):
        self.camera = camera
        self.socket = socket

    def publish_image(self, text, mode, head_tilt=-1):
        image, depth, points = self.camera.capture_image()
        camera_pose = self.camera.robot.head.get_pose_in_base_coords()

        rotated_image = np.rot90(image, k=-1)
        rotated_depth = np.rot90(depth, k=-1)
        rotated_point = np.rot90(points, k=-1)
        PILImage.fromarray(rotated_image).save("./test_rgb.png")
        np.save("./test_depth", rotated_depth)

        ## Send RGB, depth and camera intrinsics data
        send_array(self.socket, rotated_image)
        print(self.socket.recv_string())
        send_array(self.socket, rotated_depth)
        print(self.socket.recv_string()) 
        send_array(self.socket, np.array([self.camera.fy, self.camera.fx, self.camera.cy, self.camera.cx, int(head_tilt*100)]))
        print(self.socket.recv_string())

        ## Sending Object text and Manipulation mode
        self.socket.send_string(text)
        print(self.socket.recv_string())
        self.socket.send_string(mode)
        print(self.socket.recv_string())

        ## Waiting for the base and camera transforms to center the object vertically and horizontally
        self.socket.send_string("Waiting for gripper pose/ base and head trans from workstation")
        translation = recv_array(self.socket)
        self.socket.send_string("translation received by robot")
        rotation = recv_array(self.socket)
        self.socket.send_string("rotation received by robot")
        add_data = recv_array(self.socket)
        self.socket.send_string(f"Additional data received robot")

        depth = add_data[0]
        cropped = add_data[1]
        retry = add_data[2]
        print(f"Additional data received - {add_data}")
        print("translation: ")
        print(translation)
        print("rotation: ")
        print(rotation)
        print(self.socket.recv_string())    
        return translation, rotation, depth, cropped, retry
