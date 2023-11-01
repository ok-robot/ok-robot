import ml as ml_root
print(ml_root.__version__)

import os

# These environment variables control where training and eval logs are written.
# You can set these in your shell profile as well.
os.environ["RUN_DIR"] = "runs"
os.environ["EVAL_RUN_DIR"] = "eval_runs"
os.environ["MODEL_DIR"] = "models"
os.environ["DATA_DIR"] = "data"

# This is used to set a constant Tensorboard port.
os.environ["TENSORBOARD_PORT"] = str(8989)

# Useful for debugging.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import ml.api as ml  # Source: https://github.com/codekansas/ml-starter

ml.configure_logging()

import sys
sys.path.append('usa')
import hydra
# Imports these files to add them to the model and task registry.
from usa.models.point2emb import Point2EmbModel, Point2EmbModelConfig
from usa.tasks.clip_sdf import ClipSdfTask

import math
import pickle as pkl
import zipfile
from pathlib import Path
from typing import Iterator

import cv2
import imageio
import matplotlib.pyplot as plt
import ml.api as ml
import numpy as np
import requests
import torch
from IPython.display import Image
from omegaconf import OmegaConf
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from torch import Tensor

from usa.planners.base import Map, Planner, get_occupancy_map_from_dataset

import zmq

from matplotlib import pyplot as plt

from localize_cf import load_everything as load_cf
from localize_usa import load_everything as load_usa
from localize_voxel_map import load_everything as load_voxel_map

from localize_cf import localize_AonB as localize_cf
from localize_usa import localize_AonB as localize_usa
from localize_voxel_map import localize_AonB as localize_voxel_map

from usa.planners.clip_sdf import AStarPlanner, GradientPlanner
from usa.planners.base import get_ground_truth_map_from_dataset

import math

def load_map(config_path, map_type = 'conservative_vlmap', path_cfg = None):
    # TODO: Load AStar planner without using USA-Net's codes

    # Using the default config, but overriding the dataset.
    config = OmegaConf.load(config_path)
    config.task.dataset_path = os.path.join("usa", config.task.dataset_path)
    # You can change this number to change the number of training steps.
    config.task.finished.max_steps = 10
    config.trainer.base_run_dir = os.path.join('usa', config.trainer.base_run_dir)
    # Loads the config objects.
    print(config)
    objs = ml.instantiate_config(config)
    # Unpacking the different components.
    model = objs.model
    task = objs.task
    optimizer = objs.optimizer
    lr_scheduler = objs.lr_scheduler
    trainer = objs.trainer
    # Runs the training loop.
    trainer.train(model, task, optimizer, lr_scheduler)

    dataset = task._dataset
    grid_planner = AStarPlanner(
       dataset=dataset,
       model=model.double(),
       task=task.double(),
        device=task._device,
        # The heuristic to use for AStar
        heuristic="euclidean",
        # The grid resolution
        resolution=0.1 if not path_cfg else path_cfg.resolution,
        # Where to store cache artifacts
        cache_dir=None,
        # Height of the floor
        floor_height=-0.9,
        # Height of the ceiling
        ceil_height=0.1,
        occ_avoid_radius = 0.3 if not path_cfg else path_cfg.occ_avoid_radius
    ).double()

    occ_avoid = 3 if not path_cfg else math.ceil((path_cfg.occ_avoid_radius) / path_cfg.resolution)
    print(occ_avoid)
    if map_type == 'conservative_vlmap':
        print("load conservative vlmap")
        obs_map = get_occupancy_map_from_dataset(dataset, 0.1, (-1.2, 0.1), conservative = True, occ_avoid = occ_avoid).grid
        grid_planner.a_star_planner.is_occ = np.expand_dims((obs_map == -1), axis = -1)
        #grid_planner.base_planner.a_star_planner.is_occ = np.expand_dims((obs_map == -1), axis = -1)
    if map_type == 'brave_vlmap':
        print("load brave vlmap")
        obs_map = get_occupancy_map_from_dataset(dataset, 0.1, (-1.2, 0.1), conservative = False, occ_avoid = occ_avoid).grid
        grid_planner.a_star_planner.is_occ = np.expand_dims(obs_map, axis = -1)
        #grid_planner.base_planner.a_star_planner.is_occ = np.expand_dims((obs_map == -1), axis = -1)
    # if map_type == 'sdf_map', don't do anything, the default map of grid planner is sdf map
    return grid_planner, dataset

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
    #buf = buffer(msg)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

@hydra.main(version_base="1.2", config_path=".", config_name="path.yaml")
def main(cfg):
    socket = load_socket(cfg.port_number)
    grid_planner, dataset = load_map(config_path = cfg.map_config, map_type = cfg.map_type, path_cfg = cfg)
    if not cfg.debug:
        if cfg.localize_type == 'cf':
            load_everything = load_cf
            localize_AonB = localize_cf
            label_model, clip_model, preprocessor, sentence_model, points_dataloader, model_type = load_everything(cfg.cf_config)
        elif cfg.localize_type == 'usa':
            load_everything = load_usa
            localize_AonB = localize_usa
            label_model, clip_model, preprocessor, points_dataloader, model_type = load_everything(cfg.usa_config, cfg.usa_weight)
        else:
            load_everything = load_voxel_map
            localize_AonB = localize_voxel_map
            voxel_pcd, clip_model, preprocessor, model_type = load_everything(cfg.cf_config)
    while True:
        if cfg.debug:
            start_x = float(input('start x'))
            start_y = float(input('start y'))
            start_xy = [start_x, start_y]
            print(start_xy)
            end_x = float(input('end x'))
            end_y = float(input('end y'))
            end_xy = [end_x,  end_y]
            print(end_xy)
            paths = grid_planner.plan(start_xy=start_xy, end_xy=end_xy, remove_line_of_sight_points = False)
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
            if cfg.localize_type == 'cf':
                end_xyz = localize_AonB(label_model, clip_model, preprocessor, 
                    sentence_model, A, B, points_dataloader, k_A = 30, k_B = 50, linguistic = model_type, vision_weight = 10.0, text_weight = 1.0)
            elif cfg.localize_type == 'usa':
                end_xyz = localize_AonB(label_model, clip_model, preprocessor, 
                    A, B, points_dataloader, k_A = 30, k_B = 50, linguistic = model_type)
            else:
                end_xyz = localize_AonB(voxel_pcd, clip_model, preprocessor, A, B, k_A = 3, k_B = 5,
                          linguistic = model_type, data_type = 'r3d')
            end_xy = end_xyz[:2]
            paths = grid_planner.plan(start_xy=start_xyt[:2], end_xy = end_xy, remove_line_of_sight_points = True)
            end_pt = grid_planner.a_star_planner.to_pt(paths[-1][:2])
            theta = paths[-1][2] if paths[-1][2] > 0 else paths[-1][2] + 2 * np.pi
            if (theta >= np.pi / 8) and (theta <= 3 * np.pi / 8):
                move_range = (grid_planner.a_star_planner.point_is_occupied(end_pt[0] - 1, end_pt[1] + 1), grid_planner.a_star_planner.point_is_occupied(end_pt[0] + 1, end_pt[1] - 1))
            elif (theta >= 3 * np.pi / 8) and (theta <= 5 * np.pi / 8):
                move_range = (grid_planner.a_star_planner.point_is_occupied(end_pt[0] - 1, end_pt[1]), grid_planner.a_star_planner.point_is_occupied(end_pt[0] + 1, end_pt[1]))
            elif (theta >= 5 * np.pi / 8) and (theta <= 7 * np.pi / 8):
                move_range = (grid_planner.a_star_planner.point_is_occupied(end_pt[0] - 1, end_pt[1] - 1), grid_planner.a_star_planner.point_is_occupied(end_pt[0] + 1, end_pt[1] + 1))
            elif (theta >= 7 * np.pi / 8) and (theta <= 9 * np.pi / 8):
                move_range = (grid_planner.a_star_planner.point_is_occupied(end_pt[0], end_pt[1] - 1), grid_planner.a_star_planner.point_is_occupied(end_pt[0], end_pt[1] + 1))
            elif (theta >= 9 * np.pi / 8) and (theta <= 11 * np.pi / 8):
                move_range = (grid_planner.a_star_planner.point_is_occupied(end_pt[0] + 1, end_pt[1] - 1), grid_planner.a_star_planner.point_is_occupied(end_pt[0] - 1, end_pt[1] + 1))
            elif (theta >= 11 * np.pi / 8) and (theta <= 13 * np.pi / 8):
                move_range = (grid_planner.a_star_planner.point_is_occupied(end_pt[0] + 1, end_pt[1]), grid_planner.a_star_planner.point_is_occupied(end_pt[0] - 1, end_pt[1]))
            elif (theta >= 13 * np.pi / 8) and (theta <= 15 * np.pi / 8):
                move_range = (grid_planner.a_star_planner.point_is_occupied(end_pt[0] + 1, end_pt[1] + 1), grid_planner.a_star_planner.point_is_occupied(end_pt[0] - 1, end_pt[1] - 1))
            else:
                move_range = (grid_planner.a_star_planner.point_is_occupied(end_pt[0], end_pt[1] + 1), grid_planner.a_star_planner.point_is_occupied(end_pt[0], end_pt[1] - 1))
            print(socket.recv_string())
            send_array(socket, paths)
            print(socket.recv_string())
            send_array(socket, [not move_range[0], not move_range[1]])
        print(paths)
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        minx, miny = grid_planner.occ_map.origin
        (ycells, xcells), resolution = grid_planner.occ_map.grid.shape, grid_planner.occ_map.resolution
        maxx, maxy = minx + xcells * resolution, miny + ycells * resolution
        xs, ys, thetas = zip(*paths)
        axes[0].imshow(grid_planner.a_star_planner.is_occ[::-1], extent=(minx, maxx, miny, maxy))
        #axes[0].imshow(grid_planner.base_planner.a_star_planner.is_occ[::-1], extent=(minx, maxx, miny, maxy))
        axes[0].plot(xs, ys, c='r')
        axes[0].scatter(start_xyt[0], start_xyt[1], s = 50, c = 'white')
        axes[0].scatter(end_xy[0], end_xy[1], s = 50, c = 'g')
        axes[0].scatter(xs, ys, c = 'cyan', s = 10)
        axes[1].imshow(get_ground_truth_map_from_dataset(dataset, 0.1, (-0.9, 0.1)).grid[::-1], extent=(minx, maxx, miny, maxy))
        axes[1].plot(xs, ys, c='r')
        axes[1].scatter(start_xyt[0], start_xyt[1], s = 50, c = 'white')
        axes[1].scatter(end_xy[0], end_xy[1], s = 50, c = 'g')
        axes[1].scatter(xs, ys, c = 'cyan', s = 10)
        fig.savefig('output_' + A + '.jpg')


if __name__ == "__main__":
    main()
