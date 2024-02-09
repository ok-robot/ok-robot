import hydra
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from omegaconf import OmegaConf
from torch import Tensor

import zmq

from matplotlib import pyplot as plt
import open3d as o3d

import sys
sys.path.append('voxel_map')

from a_star.map_util import get_ground_truth_map_from_dataset, get_occupancy_map_from_dataset
from a_star.path_planner import PathPlanner
from a_star.data_util import get_posed_rgbd_dataset
from voxel_map_localizer import VoxelMapLocalizer
from a_star.visualizations import visualize_path

import math
import os 

import sys
sys.path.append('voxel_map')

from dataloaders import (
    R3DSemanticDataset,
    OWLViTLabelledDataset,
)

def load_socket(port_number):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + str(port_number))

    return socket

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    A = np.array(A)
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(np.ascontiguousarray(A), flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def load_dataset(cfg):
    if os.path.exists(cfg.cache_path):
        if input('\nWe have found an existing semantic memory, do you want to use it? [y|n]').lower() == 'y':
            print('\n\nSemantic memory ready!\n\n')
            return torch.load(cfg.cache_path)
    print('\n\nFetching semantic memory from record3D file, might take some time ....\n\n')
    r3d_dataset = R3DSemanticDataset(cfg.dataset_path, cfg.custom_labels, subsample_freq=cfg.sample_freq)
    semantic_memory = OWLViTLabelledDataset(
        r3d_dataset,
        owl_model_name=cfg.web_models.owl,
        sam_model_type=cfg.web_models.sam,
        device=cfg.memory_load_device,
        threshold=cfg.threshold,
        subsample_prob=cfg.subsample_prob,
        visualize_results=cfg.visualize_results,
        visualization_path=cfg.visualization_path,
    )
    torch.save(semantic_memory, cfg.cache_path)
    print('\n\nSemantic memory ready!\n\n')
    return semantic_memory

@hydra.main(version_base="1.2", config_path="configs", config_name="path.yaml")
def main(cfg):
    semantic_memory = load_dataset(cfg)
    if not cfg.debug:
        socket = load_socket(cfg.port_number)
    conservative = cfg.map_type == 'conservative_vlmap'
    planner = PathPlanner(cfg.dataset_path, cfg.min_height, cfg.max_height, cfg.resolution, cfg.occ_avoid_radius, conservative)
    localizer = VoxelMapLocalizer(semantic_memory, owl_vit_config = cfg.web_models.owl device = cfg.path_planning_device)

    obstacle_map = planner.occupancy_map
    minx, miny = obstacle_map.origin
    (ycells, xcells), resolution = obstacle_map.grid.shape, obstacle_map.resolution
    maxx, maxy = minx + xcells * resolution, miny + ycells * resolution
    dataset = planner.dataset
    ground_truth_map = get_ground_truth_map_from_dataset(dataset, cfg.resolution, (cfg.min_height, cfg.max_height))

    while True:
        if cfg.debug:
            A = input("A: ")
            B = input("B: ")
            end_xyz = localizer.localize_AonB(A, B)
            end_xy = end_xyz[:2]
            visualize_path(None, end_xyz, cfg)
        else:
            start_xyt = recv_array(socket)
            print(start_xyt)
            socket.send_string('xyt received')
            print('receive text query')
            A = socket.recv_string()
            print(A)
            socket.send_string('A received')
            B = socket.recv_string()
            print(B)
            socket.send_string('B received')
            end_xyz = localizer.localize_AonB(A, B)
            end_xy = end_xyz[:2]
            paths = planner.plan(start_xy=start_xyt[:2], end_xy = end_xy, remove_line_of_sight_points = True)
            visualize_path(paths, end_xyz, cfg)
            end_pt = planner.a_star_planner.to_pt(paths[-1][:2])
            theta = paths[-1][2] if paths[-1][2] > 0 else paths[-1][2] + 2 * np.pi
            
            print(socket.recv_string())
            send_array(socket, paths)
            print(socket.recv_string())
            send_array(socket, end_xyz)
            print(paths)
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        
        if not cfg.debug:
            xs, ys, thetas = zip(*paths)
        axes[0].imshow(obstacle_map.grid[::-1], extent=(minx, maxx, miny, maxy))
        if not cfg.debug:
            axes[0].plot(xs, ys, c='r')
            axes[0].scatter(start_xyt[0], start_xyt[1], s = 50, c = 'white')
            axes[0].scatter(xs, ys, c = 'cyan', s = 10)
        axes[0].scatter(end_xyz[0], end_xyz[1], s = 50, c = 'g')
        axes[1].imshow(ground_truth_map.grid[::-1], extent=(minx, maxx, miny, maxy))
        if not cfg.debug:
            axes[1].plot(xs, ys, c='r')
            axes[1].scatter(start_xyt[0], start_xyt[1], s = 50, c = 'white')
            axes[1].scatter(xs, ys, c = 'cyan', s = 10)
        axes[1].scatter(end_xyz[0], end_xyz[1], s = 50, c = 'g')
        if not os.path.exists(cfg.save_file + '/' + A):
            os.makedirs(cfg.save_file + '/' + A )
        print(cfg.save_file + '/' + A + '/navigation_vis.jpg')
        fig.savefig(cfg.save_file + '/' + A + '/navigation_vis.jpg')


if __name__ == "__main__":
    main()
