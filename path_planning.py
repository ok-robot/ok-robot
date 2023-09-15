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

from localize_cf import localize_AonB as localize_cf
from localize_usa import localize_AonB as localize_usa

from usa.planners.clip_sdf import AStarPlanner, GradientPlanner
from usa.planners.base import get_ground_truth_map_from_dataset

def load_map(config_path, map_type = 'conservative_vlmap'):
    # Using the default config, but overriding the dataset.
    config = OmegaConf.load(config_path)
    config.task.dataset_path = os.path.join("usa", config.task.dataset_path)
    # You can change this number to change the number of training steps.
    config.task.finished.max_steps = 0
    # Loads the config objects.
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
        resolution=0.1,
        # Where to store cache artifacts
        cache_dir=None,
        # Height of the floor
        floor_height=-1,
        # Height of the ceiling
        ceil_height=0,
        occ_avoid_radius = 0.2
    ).double()

    if map_type == 'conservative_vlmap':
        obs_map = get_occupancy_map_from_dataset(dataset, 0.1, (-1, 0), conservative = True).grid
    if map_type == 'sdf map':
        obs_map = grid_planner.get_map().grid[:, :, 0]
    if map_type == 'brave_vlmap':
        obs_map = get_occupancy_map_from_dataset(dataset, 0.1, (-1, 0), conservative = False).grid
    grid_planner.a_star_planner.is_occ = np.expand_dims((obs_map == -1), axis = -1)
    return grid_planner, dataset

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

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
    grid_planner, dataset = load_map(config_path = cfg.map_config, map_type = cfg.map_type)
    if not cfg.debug:
        if cfg.localize_type == 'cf':
            load_everything = load_cf
            localize_AonB = localize_cf
            label_model, clip_model, preprocessor, sentence_model, points_dataloader, model_type = load_everything(cfg.cf_config)
        else:
            load_everything = load_usa
            localize_AonB = localize_usa
            label_model, clip_model, preprocessor, points_dataloader, model_type = load_everything(cfg.usa_config, cfg.usa_weight)
    while True:
        start_xyt = recv_array(socket)
        print(start_xyt)
        socket.send_string('start_xyt received')
        if cfg.debug:
            print('receive end xy')
            end_xy = recv_array(socket)
            print(end_xy)
            socket.send_string('end_xy received')
            paths = grid_planner.plan(start_xy=start_xyt[:2], end_xy=end_xy)
        else:
            print('receive text query')
            A = socket.recv_string()
            print(A)
            socket.send_string('A received')
            B = socket.recv_string()
            print(B)
            socket.send_string('B received')
            if cfg.localize_type == 'cf':
                end_xyz = localize_AonB(label_model, clip_model, preprocessor, 
                    sentence_model, A, B, points_dataloader, k_A = 10, k_B = 500, linguistic = model_type)
            else:
                end_xyz = localize_AonB(label_model, clip_model, preprocessor, 
                    A, B, points_dataloader, k_A = 10, k_B = 500, linguistic = model_type)
            print(end_xyz)
            end_xy = end_xyz[:2]
            paths = grid_planner.plan(start_xy=start_xyt[:2], end_xy = end_xy)
        print(socket.recv_string())
        send_array(socket, paths)
        print(paths)
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        minx, miny = grid_planner.occ_map.origin
        (ycells, xcells), resolution = grid_planner.occ_map.grid.shape, grid_planner.occ_map.resolution
        maxx, maxy = minx + xcells * resolution, miny + ycells * resolution
        xs, ys, thetas = zip(*paths)
        axes[0].imshow(grid_planner.a_star_planner.is_occ[::-1], extent=(minx, maxx, miny, maxy))
        axes[0].plot(xs, ys, c='r')
        axes[0].scatter(start_xyt[0], start_xyt[1], s = 50, c = 'white')
        axes[0].scatter(end_xy[0], end_xy[1], s = 50, c = 'g')
        axes[0].scatter(xs, ys, c = 'cyan', s = 10)
        axes[1].imshow(get_ground_truth_map_from_dataset(dataset, 0.1, (-1, 0)).grid[::-1], extent=(minx, maxx, miny, maxy))
        axes[1].plot(xs, ys, c='r')
        axes[1].scatter(start_xyt[0], start_xyt[1], s = 50, c = 'white')
        axes[1].scatter(end_xy[0], end_xy[1], s = 50, c = 'g')
        axes[1].scatter(xs, ys, c = 'cyan', s = 10)
        fig.savefig('output.jpg')


if __name__ == "__main__":
    main()
