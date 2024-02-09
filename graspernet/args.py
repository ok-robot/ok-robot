import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x1", "--x1", type=float, help = "x coordinates of tape robot stands on")
    parser.add_argument("-y1", "--y1", type=float, help = "y coordinates of tape robot stands on")
    parser.add_argument("-x2", "--x2", type=float, help = "x coordinates of tape robot faces")
    parser.add_argument("-y2", "--y2", type=float, help = "y coordinates of tape robot faces")
    # parser.add_argument("-m", "--mode",
    #                 choices = ["move", "pick", "capture", "place"], default = "move",
    #                 help = "Choose the mode of operation."
    #                         "m  -> moving about a frame of reference to a point"
    #                         "po -> Picking a object with fixed offset")
    # We use it only in testing with graper.py
    parser.add_argument("-o1", "--picking_object", 
                    help = "picking object")
    parser.add_argument("-o2", "--placing_object", 
                    help = "placing object")
    # parser.add_argument("-bf", "--base_frame",
    #                 choices = [ "gripper_camera", "top_camera", 
    #                             "gripper_fingertip_left", "gripper_fingertip_right"], default = "gripper_camera",
    #                 help = "Operating frame of reference")
    # parser.add_argument("-t", "--transform", 
    #                 action="store_true", 
    #                 help = "Boolean for transforming a input co-ordinates to another frame of reference")
    parser.add_argument("-ip", "--ip", default = "100.107.224.62", help = "Workstation IP")
    parser.add_argument("-np", "--navigation_port", default = 5555, type=int, help = "Navigation port")
    parser.add_argument("-mp", "--manipulation_port", default = 5556, type=int, help = "Manipulation port")
    # parser.add_argument("-tf", "--transform_node", 
    #                 choices = [ "gripper_camera", "top_camera", 
    #                             "gripper_fingertip_left", "gripper_fingertip_right", 
    #                             "gripper_left", "gripper_mid"],
    #                 default = "gripper_mid",
    #                 help = "Operating frame of reference")

    return parser.parse_args()