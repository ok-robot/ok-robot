#!/usr/bin/env python

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="A simple script that takes an input file as an argument")

# Add the input file argument
parser.add_argument("input_file", help="Path to the r3d input file")

# Parse the arguments
args = parser.parse_args()

# Print the input file path
print(f"{args.input_file=}")

from a_star.map_util import get_pointcloud, get_posed_rgbd_dataset

get_pointcloud(get_posed_rgbd_dataset(key = 'r3d', path = args.input_file))

print("... created pointcloud.ply.")
