import time

import zmq
import cv2
import numpy as np
import PyKDL
from PIL import Image

from robot import HelloRobot
from args import get_args
from camera import RealSenseCamera
from utils.grasper_utils import pickup, move_to_point, capture_and_process_image
from utils.communication_utils import send_array, recv_array
from global_parameters import *

X_OFFSET, Y_OFFSET, THETA_OFFSET, r2n_matrix, n2r_matrix = None, None, None, None, None

def load_offset(x1, y1, x2, y2):

    '''
        Take coordinates of two tapes: the robot stands on tape (x1, y1) and look at tape (x2, y2)
        Compute two rotation matrices r2n_matrix and n2r_matrix that can transform coordinates 
        from robot hector slam system coordinate system to record3d coordinate system and 
        from record3d coordinate system to robot hector slam system coordinate system respectively
    '''

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

    '''
        An closed loop controller to move the robot from current positions to [x, y, theta]
        - robot: StretchClient robot controller
        - xyt_goal: target [x, y, theta] we want the robot to go
    '''

    xyt_goal = np.asarray(xyt_goal)
    # Make sure thea is within [-pi, pi]
    while xyt_goal[2] < -np.pi or xyt_goal[2] > np.pi:
        xyt_goal[2] = xyt_goal[2] + 2 * np.pi if xyt_goal[2] < -np.pi else xyt_goal[2] - 2 * np.pi
    # Closed loop navigation
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

def read_input():

    '''
        The robot takes input from human, human will send text queries to robot when this function is called.
        A is the name of the object human wants robot to pick up / place on
        B can be used to specify an object close to A and serve as functionality "A near B"
        If you leave B empty, the robot will simply localize A.
    '''

    A = str(input("Enter A: "))
    print("A = ", A)
    B = str(input("Enter B: "))
    print("B = ", B)

    return A, B

def run_navigation(robot, socket, A, B):
    
    '''
        An API for running navigation. By calling this API, human will ask the robot to find objects
        specified by "A (near B)"
        - robot: StretchClient robot controller
        - socket: ZMQ socket, used for asking workstation to compute planned path
        - A: text query specifying target object
        - B: text query specifying an object close to target object that helps localization of A, set to None, if 
                you just want the robot to localize A instead of "A near B"
    '''

    # Compute start_xy of the robot
    start_xy = robot.nav.get_base_pose()
    print(start_xy)
    transformed_start_xy = r2n_matrix @ np.array([start_xy[0], start_xy[1], 1])
    start_xy[0], start_xy[1] = transformed_start_xy[0], transformed_start_xy[1]
    start_xy[2] += THETA_OFFSET
    print(start_xy)

    # Send start_xy, A, B to the workstation and receive planned paths
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

    # Transform waypoints into robot hector slam coordinate systems and let robot navigate to those waypoints
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

def run_manipulation(hello_robot, socket, text, transform_node, base_node):
    '''
        An API for running manipulation. By calling this API, human will ask the robot to pick up objects
        specified by text queries A
        - hello_robot: a wrapper for home-robot StretchClient controller
        - socoket: we use this to communicate with workstation to get estimated gripper pose
        - text: queries specifying target object
        - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
        - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
    '''

    gripper_pos = 1

    hello_robot.move_to_position(arm_pos=INIT_ARM_POS,
                                head_pan=INIT_HEAD_PAN,
                                head_tilt=INIT_HEAD_TILT,
                                gripper_pos = gripper_pos)
    time.sleep(1)
    hello_robot.move_to_position(lift_pos=INIT_LIFT_POS,
                                wrist_pitch = INIT_WRIST_PITCH,
                                wrist_roll = INIT_WRIST_ROLL,
                                wrist_yaw = INIT_WRIST_YAW)
    time.sleep(2)

    camera = RealSenseCamera(hello_robot.robot)

    rotation, translation, depth = capture_and_process_image(
        camera = camera,
        mode = 'pick',
        obj = text,
        socket = socket, 
        hello_robot = hello_robot)
    
    if rotation is None:
        return False
        
    if input('Do you want to do this manipulation? Y or N ') != 'N':
        pickup(hello_robot, rotation, translation, base_node, transform_node, gripper_depth = depth)
    
    # Shift the base back to the original point as we are certain that orginal point is navigable in navigation obstacle map
    hello_robot.move_to_position(base_trans = -hello_robot.robot.manip.get_joint_positions()[0])

    return True

def run_place(hello_robot, socket, text, transform_node, base_node):
    '''
        An API for running placing. By calling this API, human will ask the robot to place whatever it holds
        onto objects specified by text queries A
        - hello_robot: a wrapper for home-robot StretchClient controller
        - socoket: we use this to communicate with workstation to get estimated gripper pose
        - text: queries specifying target object
        - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
        - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
    '''

    camera = RealSenseCamera(hello_robot.robot)

    time.sleep(2)
    rotation, translation, _ = capture_and_process_image(
        camera = camera,
        mode = 'place',
        obj = text,
        socket = socket, 
        hello_robot = hello_robot)

    if rotation is None:
        return False
    print(rotation)

    # lift arm to the top before the robot extends the arm, prepare the pre-placing gripper pose
    hello_robot.move_to_position(lift_pos=1.05)
    time.sleep(1)
    hello_robot.move_to_position(wrist_yaw=0,
                                 wrist_pitch=0)
    time.sleep(2)

    # Placing the object
    move_to_point(hello_robot, translation, base_node, transform_node, move_mode=0)
    hello_robot.move_to_position(gripper_pos=1)

    # Lift the arm a little bit, and rotate the wrist roll of the robot in case the object attached on the gripper
    hello_robot.move_to_position(lift_pos = hello_robot.robot.manip.get_joint_positions()[1] + 0.3)
    hello_robot.move_to_position(wrist_roll = 3)
    time.sleep(1)
    hello_robot.move_to_position(wrist_roll = -3)

    # Wait for some time and shrink the arm back
    time.sleep(4)
    hello_robot.move_to_position(gripper_pos=1, 
                                lift_pos = 1.05,
                                arm_pos = 0)
    time.sleep(4)
    hello_robot.move_to_position(wrist_pitch=-1.57)
    time.sleep(1)

    # Shift the base back to the original point as we are certain that orginal point is navigable in navigation obstacle map
    hello_robot.move_to_position(base_trans = -hello_robot.robot.manip.get_joint_positions()[0])
    return True

def compute_tilt(camera_xyz, target_xyz):
    '''
        a util function for computing robot head tilts so the robot can look at the target object after navigation
        - camera_xyz: estimated (x, y, z) coordinates of camera
        - target_xyz: estimated (x, y, z) coordinates of the target object
    '''
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

def run():
    args = get_args()
    load_offset(args.x1, args.y1, args.x2, args.y2)
    
    base_node = TOP_CAMERA_NODE

    transform_node = GRIPPER_MID_NODE
    hello_robot = HelloRobot(end_link = transform_node)

    context = zmq.Context()
    nav_socket = context.socket(zmq.REQ)
    nav_socket.connect("tcp://" + args.ip + ":" + str(args.navigation_port))
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
            perform_manip = run_manipulation(hello_robot, anygrasp_socket, A, transform_node, base_node)
            if not perform_manip:
                continue

        # clear picking object
        A, B = None, None
        print('debug coordinates', hello_robot.robot.nav.get_base_pose())
        if input("You want to run navigation? Y or N") != "N":
            A, B = read_input()

            hello_robot.robot.switch_to_navigation_mode()
            hello_robot.robot.head.look_front()
            end_xyz = run_navigation(hello_robot.robot, nav_socket, A, B)
            if not end_xyz is None:
                camera_xyz = hello_robot.robot.head.get_pose()[:3, 3]
                INIT_HEAD_TILT = compute_tilt(camera_xyz, end_xyz)

        if input("You want to run place? Y or N") != 'N':
            if (A is None):
                A, _ = read_input()
            hello_robot.robot.switch_to_manipulation_mode()
            hello_robot.robot.head.look_at_ee()
            run_place(hello_robot, anygrasp_socket, A, transform_node, base_node)
        time.sleep(1)

if __name__ == '__main__':
    run()
