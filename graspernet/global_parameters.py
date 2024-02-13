# Reference nodes usedful in this project
CAMERA_NODE = "link_raised_gripper"
TOP_CAMERA_NODE = "camera_depth_optical_frame"
GRIPPER_FINGERTIP_LEFT_NODE = "link_gripper_fingertip_left"
GRIPPER_FINGER_LEFT_NODE = "link_gripper_finger_left"
GRIPPER_MID_NODE = "link_straight_gripper"
# GRIPPER_MID_NODE = "link_grasp_center"
GRIPPER_FINGERTIP_RIGHT_NODE = "link_gripper_fingertip_right"

INIT_LIFT_POS = 0.33 #
INIT_WRIST_PITCH = 0 # -1.57 # for top-down picks(tdp) and 0 for normal picks(np)
INIT_ARM_POS = 0
INIT_WRIST_ROLL = 0
INIT_WRIST_YAW = 0
INIT_HEAD_PAN = -1.53
INIT_HEAD_TILT = -0.65

## x, y, z corrections to final goal_pose_new co-ordinates
CORRECTION_X = 0.0 # (infra2) #  0.05(color) # 0.1 # 
CORRECTION_Y = -0.0 # 0.02 # -0.03 # +0.05 for np and 0.0 fot tdp
CORRECTION_Z = 0 # -0.05 # 0.025

# Segmentation variables
SAM_CHECKPOINT = "./models/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

