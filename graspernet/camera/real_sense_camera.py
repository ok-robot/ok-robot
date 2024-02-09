## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
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
        #TODO: This is actually unecessary, we are rotating the images to make sure we can make 
        # minimal changes to our older versions of codes
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

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ix = x
            self.iy = y
            print("Selected point: ({}, {})".format(self.ix, self.iy))

    def visualize_image(self):

        # Window for Depth Image
        cv2.namedWindow("Depth Image")
        cv2.setMouseCallback("Depth Image", self.click_event)

        # Window for RGB Image
        cv2.namedWindow("RGB Image")
        cv2.setMouseCallback("RGB Image", self.click_event)

        width, height = self.rgb_image.shape[:2]
        cv2.resizeWindow("RGB Image", width, height)
        cv2.resizeWindow("Depth Image", width, height)

        cv2_rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        while(1):
            rotated_rgb = cv2.rotate(cv2_rgb_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_depth = cv2.rotate(self.depth_image, cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow("RGB Image", rotated_rgb)
            cv2.imshow("Depth Image", rotated_depth/np.max(rotated_depth))
            # cv2.waitKey(1000)
            
            if self.ix is not None and self.iy is not None:
                cv2.line(rotated_rgb, (self.ix-25, self.iy-25), (self.ix + 25, self.iy + 25), (0, 0, 255), 2)
                cv2.line(rotated_rgb, (self.ix-25, self.iy+25), (self.ix + 25, self.iy - 25), (0, 0, 255), 2)
                tiy = self.iy
                self.iy = width - self.ix
                self.ix = tiy
                cv2.imshow("RGB Image", rotated_rgb)
                cv2.waitKey(3000)
                break
                # cv2.imwrite("./images/crossed_rgb.png", rgb)
                # break
            if cv2.waitKey(1000) == 27:
                break
            
        return self.ix, self.iy 

        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera = RealSenseCamera()
    camera.capture_image()
    camera.visualize_image()
