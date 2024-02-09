#import rospy
import zmq
import numpy as np
from PIL import Image as PILImage

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray,MultiArrayDimension


IMAGE_PUBLISHER_NAME = '/gopro_image'
DEPTH_PUBLISHER_NAME = '/gopro_depth'

# use zmq to send a numpy array
def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(np.ascontiguousarray(A), flags, copy=copy, track=track)

# use zmq to receive a numpy array
def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

class ImagePublisher():

    def __init__(self, camera, socket):
        self.camera = camera
        self.bridge = CvBridge()
        self.socket = socket

    def publish_image(self, text, mode, head_tilt=-1, top_down = False):
        image, depth, points = self.camera.capture_image()
        camera_pose = self.camera.robot.head.get_pose_in_base_coords()

        rotated_image = np.rot90(image, k=-1)
        rotated_depth = np.rot90(depth, k=-1)
        rotated_point = np.rot90(points, k=-1)
        PILImage.fromarray(rotated_image).save("./images/peiqi_test_rgb22.png")
        # PILImage.fromarray(rotated_depth).save("./images/peiqi_test_depth22.png")

        ## Send RGB, depth and camera intrinsics data
        send_array(self.socket, rotated_image)
        print(self.socket.recv_string())
        send_array(self.socket, rotated_depth)
        print(self.socket.recv_string()) 
        send_array(self.socket, np.array([self.camera.fy, self.camera.fx, self.camera.cy, self.camera.cx, int(head_tilt*100)]))
        print(self.socket.recv_string())

        ## Support for home-robot top-down grasping [not need for now]
        if top_down:
            send_array(self.socket, camera_pose)
            print(self.socket.recv_string())

        ## Sending Object text and Manipulation mode
        self.socket.send_string(text)
        print(self.socket.recv_string())
        self.socket.send_string(mode)
        print(self.socket.recv_string())

        ## Waiting for the base and camera transforms to center the object vertically and horizontally
        self.socket.send_string("Waiting for gripper pose/ base and head trans")
        translation = recv_array(self.socket)
        self.socket.send_string("translation received")
        rotation = recv_array(self.socket)
        self.socket.send_string("rotation received")
        add_data = recv_array(self.socket)
        self.socket.send_string(f"Additional data received")

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
