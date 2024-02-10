from robot import HelloRobot
from args import get_args
from camera import RealSenseCamera
from nodes import ImagePublisher
from utils.grasper_utils import pickup, move_to_point

import zmq
import time
import rospy
import PyKDL
from PIL import Image

def capture_and_process_image(camera, args, socket, hello_robot, INIT_HEAD_TILT, top_down = False):
    print(args.mode)
    # point selector
    if args.mode == "move":
        # Image Capturing 
        rgb_image, depth_image, points = camera.capture_image()
        h, _, _ = rgb_image.shape

        # Displaying windows for point selection
        ix, iy = camera.visualize_image()
        print(f"ix - {ix},iy - {iy}")

        # Image to world co-ordinates conversion
        sx, sy, sz = camera.pixel2d_to_point3d(ix, iy)
        point = PyKDL.Vector(sx, -sy, sz)
        print(f"x - {sx}, y - {sy}, z - {sz}")

        rotation = PyKDL.Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1)
    
        return rotation, point

    print(args.picking_object)
    if args.mode == "pick" or args.mode == "place":
        print("hello")
        if args.mode == "pick":
            obj = args.picking_object
        else:
            obj = args.placing_object

        image_publisher = ImagePublisher(camera, socket)

        # Centering the object
        head_tilt_angles = [0, -0.1, 0.1]
        tilt_retries, side_retries = 1, 0
        retry_flag = True
        head_tilt = INIT_HEAD_TILT
        head_pan = INIT_HEAD_PAN

        while(retry_flag):
            translation, rotation, depth, cropped, retry_flag = image_publisher.publish_image(obj, args.mode, head_tilt=head_tilt, top_down = top_down)

            print(f"retry flag : {retry_flag}")
            if (retry_flag == 1):
                base_trans = translation[0]
                head_tilt += (rotation[0])

                hello_robot.move_to_position(base_trans=base_trans,
                                        head_pan=head_pan,
                                        head_tilt=head_tilt)
                time.sleep(4)
            
            elif (retry_flag !=0 and side_retries == 3):
                print("Tried in all angles but couldn't succed")
                time.sleep(2)
                return None, None, None

            elif (side_retries == 2 and tilt_retries == 3):
                hello_robot.move_to_position(base_trans=0.1, head_tilt=head_tilt)
                side_retries = 3

            elif retry_flag == 2:
                if (tilt_retries == 3):
                    if (side_retries == 0):
                        hello_robot.move_to_position(base_trans=0.1, head_tilt=head_tilt)
                        side_retries = 1
                    else:
                        hello_robot.move_to_position(base_trans=-0.2, head_tilt=head_tilt)
                        side_retries = 2
                    tilt_retries = 1
                else:
                    print(f"retrying with head tilt : {head_tilt + head_tilt_angles[tilt_retries]}")
                    hello_robot.move_to_position(head_pan=head_pan,
                                            head_tilt=head_tilt + head_tilt_angles[tilt_retries])
                    tilt_retries += 1
                    time.sleep(1)

        if args.mode == "place":
            translation = PyKDL.Vector(-translation[1], -translation[0], -translation[2])

        return rotation, translation, depth

