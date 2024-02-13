"""
 This file implements get_posed_rgbd_dataset, get_xyz, iter_xyz, and a series of util functions
    in USA-Net (https://github.com/codekansas/usa) project
 Most codes are adapted from:
    1. https://github.com/codekansas/usa/blob/master/usa/planners/base.py
    2. https://github.com/codekansas/usa/blob/master/usa/tasks/datasets/posed_rgbd.py

 
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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sized, cast, Union

import more_itertools
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data.dataset import Dataset

from a_star.dataset_class import PosedRGBDItem
from a_star.dataset_class import R3DDataset

import open3d as o3d


def get_posed_rgbd_dataset(
    key: str,
    path: str
) -> Dataset[PosedRGBDItem]:
    assert key == 'home_robot' or key == 'r3d'
    # Currently we only support data from Record3D
    if key == "r3d":
        return R3DDataset(path)


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def xdiff(self) -> float:
        return self.xmax - self.xmin

    @property
    def ydiff(self) -> float:
        return self.ymax - self.ymin

    @property
    def zdiff(self) -> float:
        return self.zmax - self.zmin

    @classmethod
    def from_arr(cls, bounds) -> "Bounds":
        assert bounds.shape == (3, 2), f"Invalid bounds shape: {bounds.shape}"

        return Bounds(
            xmin=bounds[0, 0].item(),
            xmax=bounds[0, 1].item(),
            ymin=bounds[1, 0].item(),
            ymax=bounds[1, 1].item(),
            zmin=bounds[2, 0].item(),
            zmax=bounds[2, 1].item(),
        )


def get_xyz(depth: Tensor, mask: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """Returns the XYZ coordinates for a set of points.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        mask: The mask array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)
        intrinsics: The intrinsics array, with shape (B, 3, 3)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """

    (bsz, _, height, width), device, dtype = depth.shape, depth.device, intrinsics.dtype

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat_interleave(bsz, 0)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Applies intrinsics and extrinsics.
    # xyz = xyz @ intrinsics.inverse().transpose(-1, -2)
    xyz = xyz @ get_inv_intrinsics(intrinsics).transpose(-1, -2)
    xyz = xyz * depth.flatten(1).unsqueeze(-1)
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]

    # Mask out bad depth points.
    xyz = xyz.unflatten(1, (height, width))
    xyz[mask.squeeze(1)] = 0.0

    return xyz


def iter_xyz(ds: Dataset[PosedRGBDItem], desc: str, chunk_size: int = 16) -> Iterator[tuple[Tensor, Tensor]]:
    """Iterates XYZ points from the dataset.

    Args:
        ds: The dataset to iterate points from
        desc: TQDM bar description
        chunk_size: Process this many frames from the dataset at a time

    Yields:
        The XYZ coordinates, with shape (B, H, W, 3), and a mask where a value
        of True means that the XYZ coordinates should be ignored at that
        point, with shape (B, H, W)
    """

    #device = torch.device('cuda') if torch.cuda.is_available() else torch.deivce('cpu')
    ds_len = len(ds)  # type: ignore

    for inds in more_itertools.chunked(tqdm.trange(ds_len, desc=desc), chunk_size):
        depth, mask, pose, intrinsics = (
            torch.stack(ts, dim=0)
            for ts in zip(
                #*((t.to(device) for t in (i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))                *((t.to(device) for t in (i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
                *((t for t in (i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
            )
        )
        xyz = get_xyz(depth, mask, pose, intrinsics)
        yield xyz, mask.squeeze(1)

def get_pointcloud(ds: Dataset[PosedRGBDItem], file_name: str, chunk_size: int = 16, threshold:float = 0.9) -> Iterator[tuple[Tensor, Tensor]]:
    """Iterates XYZ points from the dataset.

    Args:
        ds: The dataset to iterate points from
        desc: TQDM bar description
        chunk_size: Process this many frames from the dataset at a time

    Yields:
        The XYZ coordinates, with shape (B, H, W, 3), and a mask where a value
        of True means that the XYZ coordinates should be ignored at that
        point, with shape (B, H, W)
    """

    #device = torch.device('cuda') if torch.cuda.is_available() else torch.deivce('cpu')
    ds_len = len(ds)  # type: ignore
    xyzs = []
    rgbs = []

    for inds in more_itertools.chunked(tqdm.trange(ds_len, desc='point cloud'), chunk_size):
        rgb, depth, mask, pose, intrinsics = (
            torch.stack(ts, dim=0)
            for ts in zip(
                #*((t.to(device) for t in (i.image, i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
                *((t for t in (i.image, i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
            )
        )
        rgb = rgb.permute(0, 2, 3, 1)
        xyz = get_xyz(depth, mask, pose, intrinsics)
        #mask = (~mask & (torch.rand(mask.shape, device=mask.device) > threshold))
        mask = (~mask & (torch.rand(mask.shape) > threshold))
        rgb, xyz = rgb[mask.squeeze(1)], xyz[mask.squeeze(1)]
        rgbs.append(rgb.detach().cpu())
        xyzs.append(xyz.detach().cpu())
    
    xyzs = torch.vstack(xyzs)
    rgbs = torch.vstack(rgbs)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(xyzs)
    merged_pcd.colors = o3d.utility.Vector3dVector(rgbs)
    merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)

    o3d.io.write_point_cloud(file_name, merged_downpcd)


def get_poses(ds: Dataset[PosedRGBDItem], cache_dir = None) -> np.ndarray:
    # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
    cache_loc = None if cache_dir is None else cache_dir / "poses.npy"

    if cache_loc is not None and cache_loc.is_file():
        return np.load(cache_loc)

    all_poses: list[np.ndarray] = []
    for i in tqdm.trange(len(cast(Sized, ds)), desc="Poses"):
        all_poses.append(ds[i].pose.cpu().numpy())

    poses = np.stack(all_poses)
    if cache_loc is not None:
        cache_loc.parent.mkdir(exist_ok=True, parents=True)
        np.save(cache_loc, poses)

    return poses


def get_bounds(ds: Dataset[PosedRGBDItem], cache_dir = None) -> Bounds:
    # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
    cache_loc = None if cache_dir is None else cache_dir / "bounds.npy"

    bounds = None

    if cache_loc is not None and cache_loc.is_file():
        bounds = np.load(cache_loc)
    else:
        for xyz, mask in iter_xyz(ds, "Bounds"):
            xyz_flat = xyz[~mask]
            minv_torch, maxv_torch = aminmax(xyz_flat, dim=0)
            minv, maxv = minv_torch.cpu().numpy(), maxv_torch.cpu().numpy()
            if bounds is None:
                bounds = np.stack((minv, maxv), axis=1)
            else:
                bounds[:, 0] = np.minimum(bounds[:, 0], minv)
                bounds[:, 1] = np.maximum(bounds[:, 1], maxv)
        assert bounds is not None, "No samples found"
        if cache_loc is not None:
            cache_loc.parent.mkdir(exist_ok=True, parents=True)
            np.save(cache_loc, bounds)

    assert bounds is not None, "No samples found"
    return Bounds.from_arr(bounds)


def aminmax(x: Tensor, dim = None) -> tuple[Tensor, Tensor]:
    xmin, xmax = torch.aminmax(x, dim=dim)
    return xmin, xmax

def get_inv_intrinsics(intrinsics: Tensor) -> Tensor:
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics


if __name__ == "__main__":
    get_pointcloud(get_posed_rgbd_dataset(key='r3d', path = '/data/peiqi/home-engine/LeoBedroom.r3d'))
