import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x1", "--x1", type=float, help = "x coordinates of tape robot stands on")
    parser.add_argument("-y1", "--y1", type=float, help = "y coordinates of tape robot stands on")
    parser.add_argument("-x2", "--x2", type=float, help = "x coordinates of tape robot faces")
    parser.add_argument("-y2", "--y2", type=float, help = "y coordinates of tape robot faces")
    parser.add_argument("-ip", "--ip", help = "Workstation IP")
    parser.add_argument("-np", "--navigation_port", default = 5555, type=int, help = "Navigation port")
    parser.add_argument("-mp", "--manipulation_port", default = 5556, type=int, help = "Manipulation port")

    return parser.parse_args()