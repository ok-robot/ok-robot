import open3d as o3d
import numpy as np
import os

def create_dashed_cylinder_line(points, radius=0.03, dash_length=0.06, gap_length=0.04, color=[1, 0, 0]):  # Default color red
    geometries = []
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        vec = end_point - start_point
        seg_length = np.linalg.norm(vec)
        vec_normalized = vec / seg_length
        n_dashes = int(seg_length / (dash_length + gap_length))

        for j in range(n_dashes):
            dash_start = start_point + vec_normalized * j * (dash_length + gap_length)
            dash_end = dash_start + vec_normalized * dash_length
            print(f"dash_start - {dash_start}, dash_end - {dash_end}, vec - {vec}")
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=dash_length)
            cylinder.translate((dash_start + dash_end)/2)
            cylinder.rotate(cylinder.get_rotation_matrix_from_xyz([0, np.arctan2(vec[2], np.linalg.norm(vec[:2])), np.arctan2(vec[1], vec[0])]), center=dash_start)
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

    
    if not os.path.exists('pointcloud.ply'):
        print('\nNo pointcloud.ply found, creating a new one.\n')
        from a_star.data_util import get_pointcloud, get_posed_rgbd_dataset
        get_pointcloud(get_posed_rgbd_dataset(key = 'r3d', path = cfg.dataset_path))
        print('\npointcloud.ply created.\n')

    # Example point cloud and path points (replace with your data)
    point_cloud = o3d.io.read_point_cloud("pointcloud.ply")

    if path is not None:
        path = np.array(np.array(path).tolist())
        print(path)
        start_point = path[0, :]
    end_point = np.array(end_xyz.numpy())

    if path is not None:
        path[:, 2] = cfg.min_height
        lines = create_dashed_cylinder_line(path)
    end_point[2] = (cfg.min_height + cfg.max_height)/2

    # arrows = add_arrows_to_line(lines)

    # Create the line set for the path
    # lines = [[i, i+1] for i in range(len(path)-1)]
    # line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(path),
    #                                 lines=o3d.utility.Vector2iVector(lines))

    # Create spheres for start and end points
    if path is not None:
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)

    # Set the position of the spheres
    if path is not None:
        start_sphere.translate(start_point)
    end_sphere.translate(end_point)

    # Set different colors for clarity
    # lines.paint_uniform_color([1, 0, 0])  # Red path
    if path is not None:
        start_sphere.paint_uniform_color([0, 1, 0])  # Green start
    end_sphere.paint_uniform_color([0, 0, 1])  # Blue end

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
