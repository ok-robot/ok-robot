import copy

import numpy as np
from PIL import Image
import open3d as o3d

from utils.camera import CameraParameters
from PIL import ImageDraw


def sample_points(points, sampling_rate=1):
    N = len(points)
    num_samples = int(N*sampling_rate)
    indices = np.random.choice(N, num_samples, replace=False)
    sampled_points = points[indices]
    return sampled_points, indices

def get_3d_points(cam: CameraParameters):

    xmap, ymap = np.arange(cam.depths.shape[1]), np.arange(cam.depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = cam.depths
    points_x = (xmap - cam.cx) / cam.fx * points_z
    points_y = (ymap - cam.cy) / cam.fy * points_z

    points = np.stack((points_x, points_y, points_z), axis=2)
    return points

def show_mask(mask, ax=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def draw_seg_mask(image, seg_mask, save_file=None):
    alpha = np.where(seg_mask > 0, 128, 0).astype(np.uint8)

    image_pil = copy.deepcopy(image)
    alpha_pil = Image.fromarray(alpha)
    image_pil.putalpha(alpha_pil)

    if save_file is not None:
        image_pil.save(save_file)
        print(f"Saved Segementation Mask at {save_file}")

def draw_rectangle(image, bbox, width=5):
    img_drw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    width_increase = 5
    for _ in range(width_increase):
        img_drw.rectangle([(x1, y1), (x2, y2)], outline="green")

        x1 -= 1
        y1 -= 1
        x2 += 1
        y2 += 1
    
    return img_drw

def color_grippers(grippers, max_score, min_score):
    """
        grippers    : list of grippers of form graspnetAPI grasps
        max_score   : max score of grippers
        min_score   : min score of grippers

        For debugging purpose - color the grippers according to score
    """

    for idx, gripper in enumerate(grippers):
        g = grippers[idx]
        if max_score != min_score:
            color_val = (g.score - min_score)/(max_score - min_score)
        else:
            color_val = 1
        color = [color_val, 0, 0]
        print(g.score, color)
        gripper.paint_uniform_color(color)

    return grippers

def visualize_cloud_geometries(cloud, geometries, translation = None, rotation = None, visualize = True, save_file = None):
    """
        cloud       : Point cloud of points
        grippers    : list of grippers of form graspnetAPI grasps
        visualise   : To show windows
        save_file   : Visualisation file name
    """

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    if translation is not None:
        coordinate_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        translation[2] = -translation[2]
        coordinate_frame1.translate(translation)
        coordinate_frame1.rotate(rotation)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=visualize)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.add_geometry(cloud)
    if translation is not None:
        visualizer.add_geometry(coordinate_frame1)
    visualizer.poll_events()
    visualizer.update_renderer()

    if save_file is not None:
        ## Controlling the zoom
        view_control = visualizer.get_view_control()
        zoom_scale_factor = 1.4  
        view_control.scale(zoom_scale_factor)

        visualizer.capture_screen_image(save_file, do_render = True)
        print(f"Saved screen shot visualization at {save_file}")

    if visualize:
        visualizer.add_geometry(coordinate_frame)
        visualizer.run()
    else:
        visualizer.destroy_window()    