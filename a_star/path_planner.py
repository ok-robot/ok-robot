from pathlib import Path
from typing import Type, cast

import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data.dataset import Dataset

from .map_util import Map, get_occupancy_map_from_dataset
from .astar import AStarPlanner, Heuristic
from .dataset_class import PosedRGBDItem
from .data_util import get_posed_rgbd_dataset

import math

# for testing purpose
from matplotlib import pyplot as plt

def compute_theta(cur_x, cur_y, end_x, end_y):
    theta = 0
    if end_x == cur_x and end_y >= cur_y:
        theta = np.pi / 2
    elif end_x == cur_x and end_y < cur_y:
        theta = -np.pi / 2
    else:
        theta = np.arctan((end_y - cur_y) / (end_x - cur_x))
        if end_x < cur_x:
            theta = theta + np.pi
        # move theta into [-pi, pi] range, for this function specifically, 
        # (theta -= 2 * pi) or (theta += 2 * pi) is enough
        if theta > np.pi:
            theta = theta - 2 * np.pi
        if theta < np.pi:
            theta = theta + 2 * np.pi
    return theta

class PathPlanner():
    def __init__(
        self,
        dataset_path: str,
        floor_height: float,
        ceil_height: float,
        resolution: float = 0.1,
        occ_avoid_radius: float = 0.2,
        conservative: bool = True,
        heuristic: Heuristic = 'euclidean',
        cache_dir: Path | None = None
    ) -> None:
        self.occ_avoid_radius = occ_avoid_radius
        self.resolution = resolution
        self.occ_avoid = math.ceil((self.occ_avoid_radius) / self.resolution)

        # Gets the map from the parent class.
        # for now we assume all dataset is r3d dataset, so key is set to 'r3d'
        self.dataset = get_posed_rgbd_dataset(key='r3d', path = dataset_path)
        self.occupancy_map = get_occupancy_map_from_dataset(
            self.dataset,
            resolution,
            (floor_height, ceil_height),
            occ_avoid = self.occ_avoid,
            conservative = conservative
        )

        # Initializes the AStarPlanner using the occupancy map.
        is_occ = np.expand_dims((self.occupancy_map.grid == -1), axis = -1)
        self.a_star_planner = AStarPlanner(
            is_occ=is_occ,
            origin=self.occupancy_map.origin,
            resolution=self.occupancy_map.resolution,
            heuristic=heuristic,
        )

    def plan(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        remove_line_of_sight_points: bool = True,
    ) -> list[tuple[float, float, float]]:
        end_xy, end_theta = self.get_end_xy(start_xy, end_xy)
        waypoints = self.a_star_planner.run_astar(
            start_xy=start_xy,
            end_xy=end_xy,
            remove_line_of_sight_points=remove_line_of_sight_points,
        )
        xyt_points = []
        for i in range(len(waypoints) - 1):
            theta = compute_theta(waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1])
            xyt_points.append((waypoints[i][0], waypoints[i][1], float(theta)))
        xyt_points.append((waypoints[-1][0], waypoints[-1][1], end_theta))
        return xyt_points

    def get_end_xy(self, start_xy: tuple[float, float], end_xy: tuple[float, float]):
        start_pt = self.a_star_planner.to_pt(start_xy)
        reachable_pts = self.a_star_planner.get_reachable_points(start_pt)
        reachable_pts = list(reachable_pts)
        end_pt = self.a_star_planner.to_pt(end_xy)
        # 0.3 and 0.4 are hardcoded as described in OK-Robot paper
        avoid = math.ceil((0.3 - self.occ_avoid_radius) / self.resolution)
        ideal_dis = math.floor(0.4 / self.resolution)
        inds = torch.tensor([
            self.a_star_planner.compute_s1(end_pt, reachable_pt) 
            + self.a_star_planner.compute_s2(end_pt, reachable_pt, weight = 8, ideal_dis = ideal_dis)
            + self.a_star_planner.compute_s3(reachable_pt, weight = 8, avoid = avoid)
            for reachable_pt in reachable_pts
        ])
        ind = torch.argmin(inds)
        end_pt = reachable_pts[ind]
        x, y = self.a_star_planner.to_xy(end_pt)

        theta = compute_theta(x, y, end_xy[0], end_xy[1])

        return (float(x), float(y)), float(theta)

    def is_valid_starting_point(self, xy: tuple[float, float]) -> bool:
        return self.a_star_planner.is_valid_starting_point(xy)

if __name__ == "__main__":
    planner = PathPlanner(
        dataset_path = '/data/peiqi/home-engine/LeoBedroom.r3d',
        floor_height = -0.9,
        ceil_height = 0.6
    )
    start_xy = (0, -0.5)
    end_xy = (2, -0.5)
    paths = planner.plan(start_xy = start_xy, end_xy = end_xy)
    xs, ys, thetas = zip(*paths)
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    is_occ = planner.occupancy_map
    minx, miny = is_occ.origin
    (ycells, xcells), resolution = is_occ.grid.shape, is_occ.resolution
    maxx, maxy = minx + xcells * resolution, miny + ycells * resolution
    axes.imshow(is_occ.grid[::-1], extent=(minx, maxx, miny, maxy))
    axes.plot(xs, ys, c='r')
    axes.scatter(xs, ys, c = 'cyan', s = 10)
    axes.scatter(start_xy[0], start_xy[1], s = 50, c = 'white')
    axes.scatter(end_xy[0], end_xy[1], s = 50, c = 'g')
    fig.savefig('test.jpg')