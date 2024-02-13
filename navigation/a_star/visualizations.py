import math

import open3d as o3d
import numpy as np
import os

def create_dashed_cylinder_line(points, radius=0.03, dash_length=0.07, gap_length=0.03, color=[0, 0, 1]):  # Default color red
    geometries = []
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        vec = end_point - start_point
        seg_length = np.linalg.norm(vec)
        vec_normalized = vec / seg_length
        n_dashes = math.ceil(seg_length / (dash_length + gap_length))

        for j in range(n_dashes):
            new_dash_length = min(dash_length, seg_length - (j)*(dash_length + gap_length))
            dash_start = start_point + vec_normalized * j * (new_dash_length + gap_length)
            dash_end = dash_start + vec_normalized * new_dash_length
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=new_dash_length)
            cylinder.translate((dash_start + dash_end)/2)

            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, vec)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.pi/2
            rotation_matrix = cylinder.get_rotation_matrix_from_axis_angle(rotation_axis*rotation_angle)
            cylinder.rotate(rotation_matrix, center=dash_start)

            cylinder.paint_uniform_color(color)
            geometries.append(cylinder)
    
    return geometries

def add_arrows_to_line(line_set, arrow_length=0.2, cylinder_radius=0.02, cone_radius=0.05):
    arrows = []

    for line in line_set.lines:
        start_idx, end_idx = line
        start_point = np.asarray(line_set.points[start_idx])
        end_point = np.asarray(line_set.points[end_idx])
        cylinder, cone = create_arrow_geometry(start_point, end_point, arrow_length, cylinder_radius, cone_radius)
        arrows.append(cylinder)
        arrows.append(cone)

    return arrows

def create_arrow_geometry(start_point, end_point, arrow_length=0.2, cylinder_radius=0.02, cone_radius=0.05):
    vec = end_point - start_point
    vec_norm = np.linalg.norm(vec)
    arrow_vec = vec / vec_norm * arrow_length

    # Cylinder (arrow's body)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=arrow_length)
    cylinder.translate(start_point)
    cylinder.rotate(cylinder.get_rotation_matrix_from_xyz([np.arccos(arrow_vec[2] / np.linalg.norm(arrow_vec)), 0, np.arctan2(arrow_vec[1], arrow_vec[0])]), center=start_point)

    # Cone (arrow's head)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=arrow_length)
    cone.translate(start_point + vec - arrow_vec)
    cone.rotate(cone.get_rotation_matrix_from_xyz([np.arccos(arrow_vec[2] / np.linalg.norm(arrow_vec)), 0, np.arctan2(arrow_vec[1], arrow_vec[0])]), center=start_point + vec - arrow_vec)
    
    return cylinder, cone

def visualize_path(path, end_xyz, cfg):

    
    if not os.path.exists(cfg.pointcloud_path):
        print(f'\nNo {cfg.pointcloud_path} found, creating a new one.\n')
        from a_star.data_util import get_pointcloud, get_posed_rgbd_dataset
        get_pointcloud(get_posed_rgbd_dataset(key = 'r3d', path = cfg.dataset_path), cfg.pointcloud_path)
        print(f'\n{cfg.pointcloud_path} created.\n')

    # Example point cloud and path points (replace with your data)
    point_cloud = o3d.io.read_point_cloud(cfg.pointcloud_path)

    if path is not None:
        path = np.array(np.array(path).tolist())
        print(path)
        start_point = path[0, :]
    end_point = np.array(end_xyz.numpy())

    if path is not None:
        path[:, 2] = cfg.min_height
        lines = create_dashed_cylinder_line(path)
    end_point[2] = (cfg.min_height + cfg.max_height)/2

    # Create spheres for start and end points
    if path is not None:
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)

    # Set the position of the spheres
    if path is not None:
        start_sphere.translate(start_point)
    end_sphere.translate(end_point)

    # Set different colors for clarity
    # lines.paint_uniform_color([1, 0, 0])  # Red path
    if path is not None:
        start_sphere.paint_uniform_color([0, 1, 0])  # Green start
    end_sphere.paint_uniform_color([1, 0, 0])  # Blue end

    # Visualize
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=True)
    if path is not None:
        geometries = [point_cloud, *lines, start_sphere, end_sphere]
    else:
        geometries = [point_cloud, end_sphere]
    visualizer.poll_events()
    visualizer.update_renderer()
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.run()
    visualizer.destroy_window()
