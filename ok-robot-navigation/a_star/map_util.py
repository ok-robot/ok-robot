"""
 This file implements obstacle mapping building function get_occupancy_map_from_dataset
    in USA-Net (https://github.com/codekansas/usa) project
 Most codes are adapted from:
    1. https://github.com/codekansas/usa/blob/master/usa/planners/base.py

 License:
 MIT License

 Copyright (c) 2023 Ben Bolte

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data.dataset import Dataset

from a_star.data_util import get_bounds, get_poses, iter_xyz
from a_star.dataset_class import PosedRGBDItem

import cv2


@dataclass
class Map:
    grid: np.ndarray
    resolution: float
    origin: tuple[float, float]

    def to_pt(self, xy: tuple[float, float]) -> tuple[int, int]:
        return (
            int((xy[0] - self.origin[0]) / self.resolution + 0.5),
            int((xy[1] - self.origin[1]) / self.resolution + 0.5),
        )

    def to_xy(self, pt: tuple[int, int]) -> tuple[float, float]:
        return (
            pt[0] * self.resolution + self.origin[0],
            pt[1] * self.resolution + self.origin[1],
        )

    def is_occupied(self, pt: tuple[int, int]) -> bool:
        return bool(self.grid[pt[1], pt[0]])

def get_occupancy_map_from_dataset(
    ds: Dataset[PosedRGBDItem],
    cell_size: float,
    occ_height_range: tuple[float, float],
    occ_threshold: int = 100,
    clear_around_bot_radius: float = 0.0,
    cache_dir = None,
    ignore_cached: bool = True,
    conservative: bool = True,
    occ_avoid: int = 2
) -> Map:
    """Gets the occupancy map from the given dataset.

    This employs a voting strategy to smooth out noisy points. We detect if
    there are points in some height range, and if so, we add one vote to the
    cell. We then threshold the votes to get the occupancy map.

    Args:
        ds: The dataset to get the occupancy map from.
        cell_size: The size of each cell in the occupancy map.
        occ_height_range: The height range to consider occupied.
        occ_threshold: The count threshold to consider a cell occupied.
        clear_around_bot_radius: The radius to clear around the robot.
        cache_dir: The directory to cache the occupancy map in.
        ignore_cached: Whether to ignore the cached occupancy map.
            The constructed occupancy map doesn't change for different
            models

    Returns:
        The occupancy map.
    """

    bounds = get_bounds(ds, cache_dir)
    origin = (bounds.xmin, bounds.ymin)
    resolution = cell_size

    min_height, max_height = occ_height_range
    args = (min_height, max_height, occ_threshold, clear_around_bot_radius)
    args_str = "_".join(str(a) for a in args)
    cache_loc = None if cache_dir is None else cache_dir / f"occ_map_{args_str}.npy"

    if not ignore_cached and cache_loc is not None and cache_loc.is_file():
        occ_map = np.load(cache_loc)

    else:
        xbins, ybins = int(bounds.xdiff / resolution) + 2, int(bounds.ydiff / resolution) + 2
        counts = None
        any_counts = None

        # Counts the number of points in each cell.
        with torch.no_grad():
            for xyz, mask_tensor in iter_xyz(ds, "Occupancy Map"):
                xyz = xyz[~mask_tensor]
                xy = xyz[:, :2]

                xs = ((xy[:, 0] - origin[0] + resolution / 2) / resolution).floor().long()
                ys = ((xy[:, 1] - origin[1] + resolution / 2) / resolution).floor().long()

                if counts is None:
                    counts = xy.new_zeros((ybins, xbins), dtype=torch.int32).flatten()
                if any_counts is None:
                    any_counts = xy.new_zeros((ybins, xbins), dtype=torch.int32).flatten()

                # Counts the number of occupying points in each cell.
                occ_xys = (xyz[:, 2] >= min_height) & (xyz[:, 2] <= max_height)

                if len(occ_xys) != 0:
                    occ_inds = ys[occ_xys] * xbins + xs[occ_xys]
                    counts.index_add_(0, occ_inds, torch.ones_like(xs[occ_xys], dtype=torch.int32))

                # Keeps track of the cells that have any points from anywhere.
                inds = ys * xbins + xs

                # Does the operation on CPU if the tensor is an MPS tensor.
                # This is slower but necessary because MPS doesn't support
                # `index_fill_` for some versions.
                if any_counts.device.type == "mps":
                    any_counts.copy_(any_counts.cpu().index_fill_(0, inds.cpu(), True).to(any_counts))
                else:
                    inds.clamp_(min=0, max=any_counts.numel() - 1)
                    any_counts.index_fill_(0, inds, True)

            assert counts is not None and any_counts is not None, "No points in the dataset"
            counts = counts.reshape((ybins, xbins))
            any_counts = any_counts.reshape((ybins, xbins))

            # Clears an area around the robot's poses.
            if clear_around_bot_radius > 0:
                poses = get_poses(ds, cache_dir=cache_dir)  # (T, 4, 4) array
                pose_xys = poses[:, :2, 3]
                for x, y in pose_xys:
                    x0, x1 = x - clear_around_bot_radius, x + clear_around_bot_radius
                    y0, y1 = y - clear_around_bot_radius, y + clear_around_bot_radius
                    x0, x1 = int((x0 - origin[0]) / resolution), int((x1 - origin[0]) / resolution)
                    y0, y1 = int((y0 - origin[1]) / resolution), int((y1 - origin[1]) / resolution)
                    x0, x1 = min(max(x0, 0), xbins), min(max(x1, 0), xbins)
                    y0, y1 = min(max(y0, 0), ybins), min(max(y1, 0), ybins)
                    counts[y0:y1, x0:x1] = 0
                    any_counts[y0:y1, x0:x1] = True

            occ_map = (counts >= occ_threshold)
            occ_map_copy = occ_map.cpu().numpy().copy()

            for i in range(occ_map.shape[0]):
                for j in range(occ_map.shape[1]):
                    if occ_map_copy[i, j]:
                        occ_map[max(0, i - occ_avoid): min(occ_map.shape[0] - 1, i + occ_avoid), max(0, j - occ_avoid): min(occ_map.shape[1] - 1, j + occ_avoid)] = -1
            
            if conservative:
                occ_map = (occ_map | ~any_counts).cpu().numpy()
            else:
                occ_map = occ_map.cpu().numpy()

            if cache_loc is not None:
                cache_loc.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_loc, occ_map)
    return Map(occ_map, resolution, origin)

def get_ground_truth_map_from_dataset(
    ds, cell_size, occ_height_range,
    occ_threshold: int = 100,
    clear_around_bot_radius: float = 0.0,
    cache_dir = None,
    ignore_cached: bool = True,
):
    return get_occupancy_map_from_dataset(
    ds,
    cell_size,
    occ_height_range,
    occ_threshold = occ_threshold,
    clear_around_bot_radius = clear_around_bot_radius,
    cache_dir = cache_dir,
    ignore_cached = ignore_cached,
    conservative = False,
    occ_avoid = 0)