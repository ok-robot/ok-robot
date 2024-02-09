#import stretch_body.robot
from home_robot_hw.remote import StretchClient
import numpy as np
import PyKDL
import rospy
import sys
import os
#from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model

# sys.path.append("../")
# sys.path.append(os.path.abspath("../"))
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
import math
import time
import random
import os
from utils.robot_utils import euler_to_quat, urdf_joint_to_kdl_joint, urdf_pose_to_kdl_frame, urdf_inertial_to_kdl_rbi, kdl_tree_from_urdf_model
from global_parameters import *
import global_parameters


OVERRIDE_STATES = {}

class HelloRobot:
    def __init__(
        self,
        stretch_client_urdf_file = 'hab_stretch/urdf',
        gripper_threshold = 7.0, 
        stretch_gripper_max = 0.3, 
        stretch_gripper_min = 0, 
        end_link = GRIPPER_MID_NODE
    ):
        self.STRETCH_GRIPPER_MAX = stretch_gripper_max
        self.STRETCH_GRIPPER_MIN = stretch_gripper_min
        self.urdf_path = os.path.join(stretch_client_urdf_file, 'stretch_manip_mode.urdf')
        
        self.GRIPPER_THRESHOLD = gripper_threshold

        #Initializing ROS node
        print("hello robot starting")
        self.head_joint_list = ["joint_fake", "joint_head_pan", "joint_head_tilt"]
        self.init_joint_list = ["joint_fake","joint_lift","3","2","1" ,"0","joint_wrist_yaw","joint_wrist_pitch","joint_wrist_roll", "joint_gripper_finger_left"]

        # end_link is the frame of reference node 
        # Ex: link_raised_gripper -> camera frame of reference and
        # link_gripper_finger_left -> left gripper finger tip frame of refernce
        self.end_link = end_link 
        self.set_end_link(end_link)
        
        self.robot = StretchClient(urdf_path = stretch_client_urdf_file)
        self.robot.switch_to_manipulation_mode()
        time.sleep(2)

        # Constraining the robots movement
        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        # Joint dictionary for Kinematics
        self.setup_kdl()
        self.initialize_home_params()


    def setup_kdl(self):
        self.joints = {'joint_fake':0}
        self.head_joints = {'joint_fake':0}
        
        robot_model = URDF.from_xml_file(self.urdf_path)
        self.kdl_tree = kdl_tree_from_urdf_model(robot_model)
        self.arm_chain = self.kdl_tree.getChain('base_link', self.end_link)
        self.joint_array = PyKDL.JntArray(self.arm_chain.getNrOfJoints())

        # Forward kinematics
        self.fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self.arm_chain)
        # Inverse Kinematics
        self.ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self.arm_chain)
        self.ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self.arm_chain, self.fk_p_kdl, self.ik_v_kdl) 
    

    def set_end_link(self, link):
        if link == GRIPPER_FINGERTIP_LEFT_NODE or GRIPPER_FINGERTIP_RIGHT_NODE:
            self.joint_list = self.init_joint_list
        else:
            self.joint_list = self.init_joint_list[:-1]


    def initialize_home_params(self, home_lift = 0.43, home_arm = 0.02, home_base = 0.0, home_wrist_yaw = 0.0, home_wrist_pitch = 0.0, home_wrist_roll = 0.0, home_gripper = 1):
        self.home_lift = home_lift
        self.home_arm = home_arm
        self.home_wrist_yaw = home_wrist_yaw
        self.home_wrist_pitch = home_wrist_pitch
        self.home_wrist_roll = home_wrist_roll
        self.home_gripper = home_gripper
        self.home_base = home_base


    def home(self):
        self.move_to_position(self.home_lift, self.home_arm, self.home_base, self.home_wrist_yaw, self.home_wrist_pitch, self.home_wrist_roll, self.home_gripper)


    def get_joints(self):

        ## Names of all 13 joints
        joint_names = self.init_joint_list + ["joint_gripper_finger_right"] + self.head_joint_list[1:]
        self.updateJoints()
        joint_values = list(self.joints.values()) + [0] + list(self.head_joints.values())[1:]

        return joint_names, joint_values

    def move_to_position(self, lift_pos = None, arm_pos = None, base_trans = 0.0, wrist_yaw = None, wrist_pitch = None, wrist_roll = None, gripper_pos = None, base_theta = None, head_tilt = None, head_pan = None):

        if base_theta is not None:
            self.robot.nav.navigate_to([0, 0, base_theta])
            return
            
        target_state = self.robot.manip.get_joint_positions()
        if not gripper_pos is None:
            self.CURRENT_STATE = gripper_pos*(self.STRETCH_GRIPPER_MAX-self.STRETCH_GRIPPER_MIN)+self.STRETCH_GRIPPER_MIN
            self.robot.manip.move_gripper(self.CURRENT_STATE)
        if not arm_pos is None:
            target_state[2] = arm_pos
        if not lift_pos is None:
            target_state[1] = lift_pos
        if base_trans is None:
            base_trans = 0
        
        target_state[0] = base_trans + target_state[0]
        if not wrist_yaw is None:
            target_state[3] = wrist_yaw
        if not wrist_pitch is None:
            target_state[4] = min(wrist_pitch, 0.1)
        if not wrist_roll is None:
            target_state[5] = wrist_roll    
        
        velocities = [1]*8
        velocities[5:] = [0.01, 0.01, 0.01, 0.01]
        velocities[0] = 0.01
        self.robot.manip.goto_joint_positions(target_state, relative = False)
        # self.robot.manip.goto(target_state)
        target_head_pan, target_head_tilt = self.robot.head.get_pan_tilt()
        if not head_tilt is None:
            target_head_tilt = head_tilt
        if not head_pan is None:
            target_head_pan = head_pan
        self.robot.head.set_pan_tilt(tilt = target_head_tilt, pan = target_head_pan)
        time.sleep(0.7)

    def pickup(self, depth):
        next_gripper_pos = 0.25
        while True:
            self.robot.manip.move_gripper(next_gripper_pos)
            # time.sleep(2)

            curr_gripper_pose = self.robot.manip.get_gripper_position()
            if next_gripper_pos == -0.2 or (curr_gripper_pose > next_gripper_pos + 0.01):
                break
            
            # Open loop grippper closing 
            if next_gripper_pos > 0:
                next_gripper_pos -= 0.05
            else: 
                next_gripper_pos = -0.2

        # move up
        # target_state = self.robot.manip.get_joint_positions()
        # target_state[1] = target_state[1] + 0.1
        # self.robot.manip.goto_joint_positions(target_state)
        # time.sleep(2)

    def updateJoints(self):
        #Update the joint state values in 'self.joints' using hellorobot api calls
        state = self.robot.manip.get_joint_positions()
        origin_dist = state[0]
        
        self.joints['joint_fake'] = origin_dist
        self.joints['joint_lift'] = state[1]
        armPos = state[2]
        self.joints['3'] = armPos / 4.0
        self.joints['2'] = armPos / 4.0
        self.joints['1'] = armPos / 4.0
        self.joints['0'] = armPos / 4.0
        self.joints['joint_wrist_yaw'] = state[3]
        self.joints['joint_wrist_roll'] = state[5]
        self.joints['joint_wrist_pitch'] = OVERRIDE_STATES.get('wrist_pitch', state[4])

        # self.joints['joint_gripper_finger_left'] = self.robot.end_of_arm.status['stretch_gripper']['pos'] * (0.6/3.4) 
        self.joints['joint_gripper_finger_left'] = 0
        # print("gripper pos - ", self.robot.end_of_arm.status['stretch_gripper']['pos'])

        # Head Joints
        pan, tilt = self.robot.head.get_pan_tilt()
        self.head_joints['joint_fake'] = origin_dist
        self.head_joints['joint_head_pan'] = pan
        self.head_joints['joint_head_tilt'] = tilt

    # following function is used to move the robot to a desired joint configuration 
    def move_to_joints(self, joints, gripper, mode=0, velocities = None):
        state = self.robot.manip.get_joint_positions()
        joints['joint_wrist_pitch'] = (joints['joint_wrist_pitch'] + 1.57) % 3.14 - 1.57
        joints['joint_wrist_yaw'] = (joints['joint_wrist_yaw'] + 1.57) % 3.14 - 1.57
        joints['joint_wrist_roll'] = (joints['joint_wrist_roll'] + 1.57) % 3.14 - 1.57
        joints['joint_wrist_pitch'] = self.clamp(joints['joint_wrist_pitch'], -1.57, 0.1)
        target_state = [
            joints['joint_fake'], 
            joints['joint_lift'],
            joints['3'] + 
            joints['2'] + 
            joints['1'] + 
            joints['0'],
            joints['joint_wrist_yaw'],
            joints['joint_wrist_pitch'],
            joints['joint_wrist_roll']]
        
        
        print(f"wrist pitch -{joints['joint_wrist_pitch']}")
        # print(f"velocites: {velocities}")
        if mode == 1:
            # Moving only the lift
            target1 = [0 for _ in range(6)]
            target1[1] = target_state[1] - state[1]
            #self.robot.manip.goto_joint_positions(target1, velocities, relative=True)
            self.robot.manip.goto_joint_positions(target1, relative=True)
            time.sleep(0.7)

        elif (mode == 2):
            # Moving base first
            target1 = [0 for _ in range(6)]
            target1[0] = target_state[0]
            self.robot.manip.goto_joint_positions(target1)
            time.sleep(0.7)

            # Then move lift
            target1 = [0 for _ in range(6)]
            target1[1] = target_state[1] - state[1]
            self.robot.manip.goto_joint_positions(target1, relative=True)
            time.sleep(0.7)

        print(f"current state {state}")
        print(f"target state {target_state}")
        self.robot.manip.goto_joint_positions(target_state)
        # self.robot.manip.goto_joint_positions(target_state)
        # self.robot.manip.goto(target_state, velocities)
        time.sleep(0.7)

        #NOTE: belwo code is to fix the pitch drift issue in current hello-robot. Remove it if there is no pitch drift issue
        OVERRIDE_STATES['wrist_pitch'] = joints['joint_wrist_pitch']
    
    def get_joint_transform(self, node1, node2):
        '''
            This function takes two nodes from a robot URDF file as input and 
            outputs the coordinate frame of node2 relative to the coordinate frame of node1.

            Mainly used for transforming co-ordinates from camera frame to gripper frame.

            It assumes that both node1 and node2 are part of the same branch in the robot tree,
            and node2 appears after node1.
        '''

        # Intializing chain -> maintains list of nodes from base link to corresponding nodes
        chain1 = self.kdl_tree.getChain('base_link', node1)
        chain2 = self.kdl_tree.getChain('base_link', node2)

        # Intializing corresponding joint array and forward chain solvers
        joint_array1 = PyKDL.JntArray(chain1.getNrOfJoints())
        joint_array2 = PyKDL.JntArray(chain2.getNrOfJoints())
        # print(chain1.getNrOfJoints(), chain2.getNrOfJoints())

        fk_p_kdl1 = PyKDL.ChainFkSolverPos_recursive(chain1)
        fk_p_kdl2 = PyKDL.ChainFkSolverPos_recursive(chain2)

        # caluculating current robot joint positions in self.joints
        self.updateJoints()

        if node1 == TOP_CAMERA_NODE:
            ref_joints1 = self.head_joints
            ref_joint1_list = self.head_joint_list
        else:
            ref_joints1 = self.joints
            ref_joint1_list = self.joint_list
            
        # Updating the joint arrays from self.joints
        for joint_index in range(joint_array1.rows()):
            joint_array1[joint_index] = ref_joints1[ref_joint1_list[joint_index]]
            # print(f"{ref_joint1_list[joint_index]} - {joint_array1[joint_index]}")

        for joint_index in range(joint_array2.rows()):
            joint_array2[joint_index] = self.joints[self.joint_list[joint_index]]
            # print(f"{self.joint_list[joint_index]} - {joint_array2[joint_index]}")
            
        # Intializing frames for corresponding to nodes
        frame1 = PyKDL.Frame()
        frame2 = PyKDL.Frame()

        # Calculating current frames of nodes
        fk_p_kdl1.JntToCart(joint_array1, frame1)
        fk_p_kdl2.JntToCart(joint_array2, frame2)
        
        # This allows to transform a point in frame1 to frame2
        frame_transform = frame2.Inverse() * frame1

        return frame_transform, frame2, frame1
    
    def move_to_pose(self, translation_tensor, rotational_tensor, gripper, move_mode=0, velocities=None):
        

        translation = [translation_tensor[0], translation_tensor[1], translation_tensor[2]]
        rotation = rotational_tensor
        print('translation and rotation', translation_tensor, rotational_tensor)
        
        # move logic
        self.updateJoints()
        
        for joint_index in range(self.joint_array.rows()):
            self.joint_array[joint_index] = self.joints[self.joint_list[joint_index]]
            # print(f"{joint_index} - {self.joint_array[joint_index]}")
        print("\n\n")

        curr_pose = PyKDL.Frame()
        del_pose = PyKDL.Frame()
        self.fk_p_kdl.JntToCart(self.joint_array, curr_pose)

        rot_matrix = R.from_euler('xyz', rotation, degrees=False).as_matrix()
        print("rot_matrix", rot_matrix)


        del_rot = PyKDL.Rotation(PyKDL.Vector(rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0]),
                                  PyKDL.Vector(rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1]),
                                  PyKDL.Vector(rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]))
        # del_rot = rotation_matrix
        del_trans = PyKDL.Vector(translation[0], translation[1], translation[2])
        del_pose.M = del_rot
        del_pose.p = del_trans
        goal_pose_new = curr_pose*del_pose    

        seed_array = PyKDL.JntArray(self.arm_chain.getNrOfJoints())
        self.ik_p_kdl.CartToJnt(seed_array, goal_pose_new, self.joint_array)

        ik_joints = {}
        for joint_index in range(self.joint_array.rows()):
            ik_joints[self.joint_list[joint_index]] = self.joint_array[joint_index]
        # print("ik joints - ", ik_joints)

        self.move_to_joints(ik_joints, gripper, move_mode, velocities)
        # time.sleep(2)

        self.updateJoints()
        for joint_index in range(self.joint_array.rows()):
            self.joint_array[joint_index] = self.joints[self.joint_list[joint_index]]
            # print(f"{joint_index} - {self.joint_array[joint_index]}")
        



