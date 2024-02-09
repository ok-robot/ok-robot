import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np

#from home_robot_hw.remote import StretchClient
from robot import HelloRobot
#import stretch_body.robot
import zmq
import time

#from global_parameters import *
#import global_parameters
from args import get_args
from camera import RealSenseCamera
from utils import potrait_to_landscape, segment_point_cloud, plane_detection, display_image_and_point

import rospy
import cv2
import numpy as np
import sys
import PyKDL
from PIL import Image

from multiprocessing import Process

from utils.grasper_utils import pickup, move_to_point
from grasper import capture_and_process_image

POS_TOL = 0.1
YAW_TOL = 0.2

GRIPPER_MID_NODE = "link_straight_gripper"
TOP_CAMERA_NODE = "camera_depth_optical_frame"

INIT_LIFT_POS = 0.33 
INIT_WRIST_PITCH = -1.57 
INIT_ARM_POS = 0
INIT_WRIST_ROLL = 0
INIT_WRIST_YAW = 0
INIT_HEAD_PAN = -1.53
INIT_HEAD_TILT = -0.65

X_OFFSET, Y_OFFSET, THETA_OFFSET, r2n_matrix, n2r_matrix = None, None, None, None, None

def load_offset(x1, y1, x2, y2):
    global X_OFFSET, Y_OFFSET, THETA_OFFSET, r2n_matrix, n2r_matrix
    X_OFFSET = x1
    Y_OFFSET = y1
    # x1 = X_OFFSET, x2 = another x
    THETA_OFFSET =  np.arctan2((y2 - y1), (x2 - x1))

    print(f"offsets - {X_OFFSET}, {Y_OFFSET}, {THETA_OFFSET}")
    r2n_matrix = \
        np.array([
            [1, 0, X_OFFSET],
            [0, 1, Y_OFFSET],
            [0, 0, 1]
        ]) @ \
        np.array([
            [np.cos(THETA_OFFSET), -np.sin(THETA_OFFSET), 0],
            [np.sin(THETA_OFFSET), np.cos(THETA_OFFSET), 0],
            [0, 0, 1]
        ])

    n2r_matrix = \
        np.array([
            [np.cos(THETA_OFFSET), np.sin(THETA_OFFSET), 0],
            [-np.sin(THETA_OFFSET), np.cos(THETA_OFFSET), 0],
            [0, 0, 1]
        ]) @ \
        np.array([
            [1, 0, -X_OFFSET],
            [0, 1, -Y_OFFSET],
            [0, 0, 1]
        ])

def navigate(robot, xyt_goal):
    xyt_goal = np.asarray(xyt_goal)
    while xyt_goal[2] < -np.pi or xyt_goal[2] > np.pi:
        xyt_goal[2] = xyt_goal[2] + 2 * np.pi if xyt_goal[2] < -np.pi else xyt_goal[2] - 2 * np.pi
    while True:
        robot.nav.navigate_to(xyt_goal, blocking = False)
        xyt_curr = robot.nav.get_base_pose()
        print("The robot currently loactes at " + str(xyt_curr))
        if np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL) and \
                (np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL)\
                 or np.allclose(xyt_curr[2], xyt_goal[2] + np.pi * 2, atol=YAW_TOL)\
                 or np.allclose(xyt_curr[2], xyt_goal[2] - np.pi * 2, atol=YAW_TOL)):
            print("The robot is finally at " + str(xyt_goal))
            break

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    paths = data.data
    i = 0
    while i < len(paths):
        x = -paths[i]
        y = paths[i + 1]
        navigate(robot, np.array([x, y, 0]))
        i += 2

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    A = np.array(A)
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

def read_input():
    A = str(input("Enter A: "))
    print("A = ", A)
    B = str(input("Enter B: "))
    print("B = ", B)

    return A, B

def run_navigation(robot, socket, A, B):
    start_xy = robot.nav.get_base_pose()
    print(start_xy)
    transformed_start_xy = r2n_matrix @ np.array([start_xy[0], start_xy[1], 1])
    start_xy[0], start_xy[1] = transformed_start_xy[0], transformed_start_xy[1]
    start_xy[2] += THETA_OFFSET
    print(start_xy)

    send_array(socket, start_xy)
    print(socket.recv_string())
    socket.send_string(A)
    print(socket.recv_string())
    socket.send_string(B)
    print(socket.recv_string())
    socket.send_string("Waiting for path")
    paths = recv_array(socket)
    print(paths)
    socket.send_string("Path received")
    end_xyz = recv_array(socket)
    z = end_xyz[2]
    end_xyz = (n2r_matrix @ np.array([end_xyz[0], end_xyz[1], 1]))
    end_xyz[2] = z

    if input("Start navigation? Y or N ") == 'N':
        return None
    
    # Let the robot run faster
    robot.nav.set_velocity(v = 25, w = 20)

    final_paths = []
    for path in paths:
        transformed_path = n2r_matrix @ np.array([path[0], path[1], 1])
        transformed_path[2] = path[2] - THETA_OFFSET
        print(transformed_path)
        final_paths.append(transformed_path)
        navigate(robot, transformed_path)
    xyt = robot.nav.get_base_pose()
    xyt[2] = xyt[2] + np.pi / 2
    navigate(robot, xyt)
    return end_xyz

def run_manipulation(args, hello_robot, socket, text, transform_node, base_node, move_range = [False, False], top_down = False):
    
    gripper_pos = 1

    print(INIT_ARM_POS, INIT_WRIST_PITCH, INIT_WRIST_ROLL, INIT_WRIST_YAW, gripper_pos)
    #print("coordinates - ", print(hello_robot.robot.nav.get_base_pose()))
    hello_robot.move_to_position(arm_pos=INIT_ARM_POS,
                                head_pan=INIT_HEAD_PAN,
                                head_tilt=INIT_HEAD_TILT,
                                gripper_pos = gripper_pos)
    time.sleep(1)
    #print("coordinates - ", print(hello_robot.robot.nav.get_base_pose()))
    hello_robot.move_to_position(lift_pos=INIT_LIFT_POS,
                                #wrist_pitch = global_parameters.INIT_WRIST_PITCH,
                                wrist_pitch = INIT_WRIST_PITCH,
                                wrist_roll = INIT_WRIST_ROLL,
                                wrist_yaw = INIT_WRIST_YAW)
    time.sleep(2)
    #print("coordinates - ", print(hello_robot.robot.nav.get_base_pose()))

    camera = RealSenseCamera(hello_robot.robot)

    args.mode = 'pick'
    args.picking_object = text
    rotation, translation, depth = capture_and_process_image(camera, args, socket, hello_robot, INIT_HEAD_TILT, top_down = top_down)
    
    #print("coordinates - ", print(hello_robot.robot.nav.get_base_pose()))
    if input('Do you want to do this manipulation? Y or N ') != 'N':
        pickup(hello_robot, rotation, translation, base_node, transform_node, top_down = top_down, gripper_depth = depth)
    
    #print("coordinates - ", print(hello_robot.robot.nav.get_base_pose()))
    # Shift back to the original point
    hello_robot.move_to_position(base_trans = -hello_robot.robot.manip.get_joint_positions()[0])
    
    #print("coordinates - ", print(hello_robot.robot.nav.get_base_pose()))

def run_place(args, hello_robot, socket, text, transform_node, base_node, move_range = [False, False], top_down = False):

    camera = RealSenseCamera(hello_robot.robot)

    args.mode = 'place'
    args.placing_object = text
    time.sleep(2)
    rotation, translation, _ = capture_and_process_image(camera, args, socket, hello_robot, INIT_HEAD_TILT, top_down = top_down)
    print(rotation)
    hello_robot.move_to_position(lift_pos=1.1)
    time.sleep(1)
    hello_robot.move_to_position(wrist_yaw=0,
                                 wrist_pitch=0)
    time.sleep(1)
    # hello_robot.move_to_position(wrist_yaw=0)
    #hello_robot.move_to_position(lift_pos=1.1)
    # hello_robot.move_to_position(wrist_pitch=0)
    time.sleep(1)
    move_to_point(hello_robot, translation, base_node, transform_node, move_mode=0)
    #move_to_point(hello_robot, translation, base_node, transform_node, move_mode=0)
    #time.sleep(4)
    #move_to_point(hello_robot, translation, base_node, transform_node, move_mode=0, pitch_rotation=-1.57)
    hello_robot.move_to_position(gripper_pos=1)
    hello_robot.move_to_position(lift_pos = hello_robot.robot.manip.get_joint_positions()[1] + 0.3)
    hello_robot.move_to_position(wrist_roll = 3)
    time.sleep(1)
    hello_robot.move_to_position(wrist_roll = -3)
    time.sleep(4)
    hello_robot.move_to_position(gripper_pos=1, 
                                lift_pos = 1.1,
                                arm_pos = 0)
    time.sleep(4)
    #hello_robot.move_to_position(wrist_pitch=-1.57, arm_pos = 0)
    #if abs(robot.robot.manip.get_joint_positions()[3] - 2.5) > 0.1:
    #    hello_robot.move_to_position(wrist_yaw  = - 2.5)
    # hello_robot.move_to_position(lift_pos=1.1)
    # hello_robot.move_to_position(arm_pos=0)
    hello_robot.move_to_position(wrist_pitch=-1.57)
    time.sleep(1)
    hello_robot.move_to_position(base_trans = -hello_robot.robot.manip.get_joint_positions()[0])

def compute_tilt(camera_xyz, target_xyz):
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

def run():
    # hello_robot = HelloRobot()
    args = get_args()
    load_offset(args.x1, args.y1, args.x2, args.y2)
    
    base_node = TOP_CAMERA_NODE

    transform_node = GRIPPER_MID_NODE
    hello_robot = HelloRobot(end_link = transform_node)


    #INIT_WRIST_PITCH = -1.57
    #global_parameters.INIT_WRIST_PITCH = -1.57
    #camera = RealSenseCamera(hello_robot.robot)
    #image_publisher = ImagePublisher(camera)

    context = zmq.Context()
    nav_socket = context.socket(zmq.REQ)
    nav_socket.connect("tcp://" + args.ip + ":" + str(args.navigation_port))
    #nav_socket.connect("tcp://172.24.71.253:5555")
    anygrasp_socket = context.socket(zmq.REQ)
    anygrasp_socket.connect("tcp://" + args.ip + ":" + str(args.manipulation_port))

    while True:
        A = None
        if input("You want to run navigation? Y or N") != "N":
            A, B = read_input()

            hello_robot.robot.switch_to_navigation_mode()
            hello_robot.robot.move_to_post_nav_posture()
            hello_robot.robot.head.look_front()
            end_xyz = run_navigation(hello_robot.robot, nav_socket, A, B)
            if not end_xyz is None:
                camera_xyz = hello_robot.robot.head.get_pose()[:3, 3]
                INIT_HEAD_TILT = compute_tilt(camera_xyz, end_xyz)

        print('debug coordinates', hello_robot.robot.nav.get_base_pose())
        if input("You want to run manipulation? Y or N ") != 'N':
            if (A is None):
                A, _ = read_input()
        
            hello_robot.robot.switch_to_manipulation_mode()
            hello_robot.robot.head.look_at_ee()
            run_manipulation(args, hello_robot, anygrasp_socket, A, transform_node, base_node)

        # clear picking object
        A, B = None, None
        print('debug coordinates', hello_robot.robot.nav.get_base_pose())
        if input("You want to run navigation? Y or N") != "N":
            A, B = read_input()

            hello_robot.robot.switch_to_navigation_mode()
            # hello_robot.robot.move_to_post_nav_posture()
            hello_robot.robot.head.look_front()
            end_xyz = run_navigation(hello_robot.robot, nav_socket, A, B)
            if not end_xyz is None:
                camera_xyz = hello_robot.robot.head.get_pose()[:3, 3]
                INIT_HEAD_TILT = compute_tilt(camera_xyz, end_xyz)

        if input("You want to run place? Y or N") != 'N':
            if (A is None):
                A, _ = read_input()
            hello_robot.robot.switch_to_manipulation_mode()
            # hello_robot.robot.move_to_manip_posture()
            hello_robot.robot.head.look_at_ee()
            run_place(args, hello_robot, anygrasp_socket, A, transform_node, base_node)
        time.sleep(1)

if __name__ == '__main__':
    run()
