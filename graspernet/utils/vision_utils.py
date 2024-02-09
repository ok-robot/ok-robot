import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

#from segment import segment_image

def plane_detection(pcd, vis=False):
    """
        Outputs center of the plane computed from the point cloud
    """

    # Estimating normals
    if not pcd.has_normals():
        pcd.estimate_normals()

    # using all defaults for computing plane boudning boxes
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0.01,
        min_num_points=180,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))

    print("Detected {} patches".format(len(oboxes)))

    # print(oboxes)

    if vis:
        geometries = []
        geometries.append(pcd)
        o3d.visualization.draw_geometries(geometries)

        for obox in oboxes:
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
            mesh.paint_uniform_color(obox.color)
            geometries.append(mesh)
            geometries.append(obox)

            # print(obox.center)
            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(np.asanyarray([obox.center]))
            # pcd1.colors = o3d.utility.Vector3dVector([[1,0,0]])
            # geometries.append(pcd1)
        

        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # convex_hull = pcd.get_minimal_oriented_bounding_box()
        # print(convex_hull)
        # bounding_sphere = pcd.get_axis_aligned_bounding_box()
        # geometries.append(coordinate_frame)
        # geometries.append(convex_hull)

        o3d.visualization.draw_geometries(geometries)
    
    return oboxes[0].center

def segment_point_cloud(rgb_image, depth_image, points, object_name):

    predictions = segment_image(object_name)

    # extracting segment mask and bounding boxes 
    seg_mask = predictions['instances'].pred_masks[0].numpy()
    bbox = predictions['instances'].pred_boxes[0].tensor.numpy().astype(int)

    # Croppping images, segment mask and point cloud according to bounding box
    xmin, ymin, xmax, ymax = bbox[0][0],  bbox[0][1], bbox[0][2] + 1, bbox[0][3] + 1

    # cropping out edges
    box_h, box_w = ymax - ymin, xmax - xmin
    print(f"bbox - [{xmin, ymin}], [{xmax, ymax}]")
    print(f"bbox height - {box_h}, width - {box_w}")
    print(f"Total bbox pixels - {box_h*box_w}, and seg pixels {seg_mask.sum()}")
    # xmin = xmin + int(0.02*box_w)
    # xmax = xmax - int(0.02*box_w)
    # ymin = ymin + int(0.02*box_h)
    # ymax = ymax - int(0.02*box_h)

    crop_rgb_image = rgb_image[ymin:ymax, xmin:xmax]
    crop_depth_image = depth_image[ymin:ymax, xmin:xmax]
    crop_points = points[ymin:ymax, xmin:xmax]
    crop_seg_mask = seg_mask[ymin:ymax, xmin:xmax]

    plt.imshow(crop_rgb_image)
    plt.title("Cropped rgb image")
    plt.savefig("cropped_seg.png")
    plt.pause(5)
    plt.close()

    # Zeroing the depth for non object pixels in bounding box
    seg_3d_mask = np.dstack((crop_seg_mask, crop_seg_mask, crop_seg_mask))
    crop_points1 = np.zeros(crop_points.shape)
    crop_points1[:,:,0:2] = crop_points[:,:,0:2]
    crop_points = np.where(np.invert(seg_3d_mask), 0, crop_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(crop_points.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector((crop_rgb_image/255).reshape(-1, 3))

    return pcd

def potrait_to_landscape(ix, iy, w):
    temp = ix
    ix = iy
    iy = w - temp

    return ix, iy


def display_image_and_point(rgb_image, ix=None, iy=None, timer=True):
    
    if ix is not None and iy is not None:
        cv2.line(rgb_image, (ix-15, iy-15), (ix + 15, iy + 15), (255, 0, 0), 2)
        cv2.line(rgb_image, (ix-15, iy+15), (ix + 15, iy - 15), (255, 0, 0), 2)

    cv2_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("out_seg_point.png", cv2_rgb_image)

    plt.imshow(rgb_image)
    plt.title('Image')
    if timer:
        plt.pause(4)
        plt.close()
    else:
        plt.show()
        plt.close()
   








    
