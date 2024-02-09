import PyKDL
import time
import math

def move_to_point(robot, point, base_node, gripper_node, move_mode=1, pitch_rotation=0):
    rotation = PyKDL.Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1)

    dest_frame = PyKDL.Frame(rotation, point)
    transform, _, _ = robot.get_joint_transform(base_node, gripper_node)

    # Rotation from gripper frame frame to gripper frame
    transformed_frame = transform * dest_frame

    transformed_frame.p[2] -= 0.2

    robot.move_to_pose(
            [transformed_frame.p[0], transformed_frame.p[1], transformed_frame.p[2]],
            [pitch_rotation, 0, 0],
            [1],
            move_mode=move_mode
        )

def pickup(robot, rotation, translation, base_node, gripper_node, gripper_height = 0.03, gripper_depth=0.03, top_down = False):
    """
        rotation: Relative rotation of gripper pose w.r.t camera
        translation: Relative translation of gripper pose w.r.t camera
        cam2gripper_transform: Transform for 
    """
    if top_down:
        point = PyKDL.Vector(translation[1], -translation[0], translation[2])

        rotation1_bottom = PyKDL.Rotation(0.0000000, 1.0000000,  0.0000000,
                                -1.0000000,  0.0000000,  0.0000000, 
                                0.0000000,  0.0000000, 1.0000000)
    else:
        point = PyKDL.Vector(-translation[1], -translation[0], translation[2])

        # Rotation from camera frame to model frame
        rotation1_bottom = PyKDL.Rotation(0.0000000, -1.0000000,  0.0000000,
                                    -1.0000000,  0.0000000,  0.0000000, 
                                    0.0000000,  0.0000000, 1.0000000)

    # Rotation from model frame to pose frame
    rotation1 = PyKDL.Rotation(rotation[0][0], rotation[0][1], rotation[0][2],
                            rotation[1][0],  rotation[1][1], rotation[1][2],
                                rotation[2][0],  rotation[2][1], rotation[2][2])
    
    gripper_yaw = math.atan(rotation[1][0]/rotation[0][0])
    print(f"gripper_yaw - {gripper_yaw}")

    # Rotation from camera frame to pose frame
    rotation =  rotation1_bottom * rotation1

    dest_frame = PyKDL.Frame(rotation, point) 

    # Can remove base and gripper node
    cam2gripper_transform, _, _ = robot.get_joint_transform(base_node, gripper_node)
    # Rotation from gripper frame frame to gripper frame
    transformed_frame = cam2gripper_transform * dest_frame

    # Lifting the arm to high position
    robot.move_to_position(lift_pos = 1.1, head_pan = None, head_tilt = None)
    time.sleep(2)

    # Rotation for aligning gripper frame   to model pose frame
    if top_down:
        rotation2_top = PyKDL.Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1)
    else:
        rotation2_top = PyKDL.Rotation(0, 0, 1, 1, 0, 0, 0, -1, 0)

    # final Rotation of gripper to hold the objet
    final_rotation = transformed_frame.M * rotation2_top
    print(f"final rotation - {final_rotation.GetRPY()}")
    robot.move_to_pose(
            [0, 0, 0],
            [final_rotation.GetRPY()[0], final_rotation.GetRPY()[1], final_rotation.GetRPY()[2]],
            [1],
        )
    time.sleep(2)

    cam2base_transform, _, _ = robot.get_joint_transform(base_node, 'base_link')
    cam2gripper_transform, _, _ = robot.get_joint_transform(base_node, gripper_node)

    transformed_point1 = cam2gripper_transform * point
    base_point = cam2base_transform * point
    print(f"transformed point1 - {transformed_point1}")

    diff_value = (0.228 - gripper_depth - gripper_height)
    transformed_point1[2] -= (diff_value)
    ref_diff = (diff_value)

    # Moving gripper to a point that is 0.2m away from the pose center in the line of gripper
    robot.move_to_pose(
        [transformed_point1.x(), transformed_point1.y(), transformed_point1.z() - 0.2],
        [0, 0, 0],
        [1],
        move_mode = 1
    )
    time.sleep(4)

    base2gripper_transform, _, _ = robot.get_joint_transform('base_link', gripper_node)
    transformed_point2 = base2gripper_transform * base_point
    print(f"transformed point2 : {transformed_point2}")
    curr_diff = transformed_point2.z()

    diff = abs(curr_diff - ref_diff)
    velocities = [1]*8
    velocities[5:] = [0.03, 0.03, 0.03, 0.03]
    velocities[0] = 0.03
    if diff > 0.08:
        dist = diff - 0.08
        robot.move_to_pose(
            [0, 0, dist],
            [0, 0, 0],
            [1]
        )
        time.sleep(2)
        base2gripper_transform, _, _ = robot.get_joint_transform('base_link', gripper_node)
        print(f"transformed point3 : {base2gripper_transform * base_point}")
        diff = diff - dist
        
    while diff > 0.01:
        dist = min(0.03, diff)
        robot.move_to_pose(
            [0, 0, dist],   
            [0, 0, 0],
            [1],
            velocities=velocities
        )
        time.sleep(2)
        base2gripper_transform, _, _ = robot.get_joint_transform('base_link', gripper_node)
        print(f"transformed point3 : {base2gripper_transform * base_point}")
        diff = diff - dist
    
    robot.pickup(abs(0))
    #robot.move_to_position(lift_pos = robot.robot.manip.get_joint_positions()[1])
    robot.move_to_position(lift_pos = 1.1)
    time.sleep(2)
    #time.sleep(5)

    # Put it down for now
    # robot.move_to_position(gripper_pos = 1)
    # robot.move_to_position(arm_pos = 0)
    # #robot.move_to_position(lift_pos = 1.1)  
    # robot.move_to_position(wrist_pitch = -1.57, 
    #                         wrist_yaw = 0,
    #                         wrist_roll = 0)

    # # Shift back to the original point
    # robot.move_to_position(base_trans = -robot.manip.get_joint_positions()[0])
    robot.move_to_position(arm_pos = 0)
    time.sleep(2)
    robot.move_to_position(wrist_pitch = 0.0, arm_pos = 0)
    time.sleep(2)
    robot.move_to_position(wrist_yaw  = 2.5, arm_pos = 0)
    time.sleep(2)
    if abs(robot.robot.manip.get_joint_positions()[3] - 2.5) > 0.1:
        robot.move_to_position(wrist_yaw  = - 2.5)
        time.sleep(1)
    # robot.move_to_position(wrist_pitch = -1.57, 
    #                         wrist_yaw = 0,
    #                         wrist_roll = 0)
    # cur_pitch = robot.robot.manip.get_joint_positions()[4]
    # for i in range(5):
    #     p = (cur_pitch * (5 - i) - 1.57 * i) / 5
    #     robot.move_to_position(wrist_pitch = p)
    # robot.move_to_position(wrist_yaw = 0)
    robot.move_to_position(lift_pos = 0.45)
    time.sleep(1)


