## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

class RealSenseCamera:
    def __init__(self, robot):
        self.robot = robot
        self.depth_scale = 0.001

        # Camera intrinsics
        intrinsics = self.robot.dpt_cam.K 
        self.fy = intrinsics[0, 0]
        self.fx = intrinsics[1, 1]
        self.cy = intrinsics[0, 2]
        self.cx = intrinsics[1, 2]
        print(self.fx, self.fy, self.cx, self.cy)

        # selected ix and iy co-ordinates
        self.ix, self.iy = None, None

    def capture_image(self, visualize=False):
        self.rgb_image, self.depth_image, self.points = self.robot.head.get_images(compute_xyz=True)
        self.rgb_image = np.rot90(self.rgb_image, k = 1)[:, :, [2, 1, 0]]
        self.depth_image = np.rot90(self.depth_image, k = 1)
        self.points = np.rot90(self.points, k = 1)

        cv2.imwrite("./images/input.jpg", np.rot90(self.rgb_image, k=-1))
        # cv2.imwrite("depth.jpg", np.rot90(self.depth_image, k=-1)/np.max(self.depth_image))
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        
        if visualize:
            fig, ax = plt.subplots(1, 2, figsize=(10,5))
            timer = fig.canvas.new_timer(interval = 5000) #creating a timer object and setting an interval of 3000 milliseconds
            timer.add_callback(lambda : plt.close())

            ax[0].imshow(np.rot90(self.rgb_image, k=-1))
            ax[0].set_title("Color Image")

            ax[1].imshow(np.rot90(self.depth_image, k=-1))
            ax[1].set_title("Depth Image")
                
            plt.savefig("./images/rgb_dpt.png")
            plt.pause(3)
            plt.close()
        
        return self.rgb_image, self.depth_image, self.points

    def pixel2d_to_point3d(self, ix, iy):
        return self.points[iy, ix][[1, 0, 2]]

if __name__ == "__main__":
    camera = RealSenseCamera()
    camera.capture_image()
