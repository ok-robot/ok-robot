# Experiment Setup
**Environment Setup:** Use Record3D to scan the environments. Recording should include: 
* All Obstacles in environments
* Complete floor where the robots can navigate
* All testing objects.
* Two tapes in the scene which will serve as a orgin for the robot.

**Tape Placement:** Follow images below to put tapes on the ground. After properly putting tapes on the ground, you should start scan the environment before moving the robot into the environment.
<p align="center">
  <img src="docs_image/How%20you%20should%20place%20tapes1.png" width="30%" height="200" style="margin-right: 10px;"/>
  <img src="docs_image/How%20you%20should%20place%20tapes2.png" width="30%" height="200"/>
</p>

**Scan the Environment:** After positioning the objects and tapes, proceed to scan the environment and save the Record3D r3d file in the `ok-robot-navigation/r3d/` folder. If you just want to try out a sample r3d file, you can use `ok-robot-navigation/r3d/sample.r3d`.

**Extract PointCloud:** Then, use the `ok-robot-navigation/get_point_cloud.py` script to extract the pointcloud of the scene, which will be stored as `ok-robot-navigation/pointcloud.ply`. 
```
python get_point_cloud.py --input_file=[your r3d file]
```
**Identify Tape Co-ordinates** Use a 3D Visualizer to determine the coordinates of the two orange tapes in the environment. Let the co-ordinates of these tapes are (x1, y1) for the first tape(tape1) and (x2, y2) for the second tape(tape2).

We recommend using CloudCompare to localize coordinates of tapes. See these images for instruction. You can download CloudCompare from [here](https://www.danielgm.net/cc/release/)
<p align="center">
  <img src="docs_image/CloudCompare step1.png" width="40%" height="200" style="margin-right: 10px;"/>
  <img src="docs_image/CloudCompare step2.png" width="40%" height="200"/>
</p>
<p align="center">
  <img src="docs_image/CloudCompare step3.png" width="40%" height="auto" style="margin-right: 10px;"/>
  <img src="docs_image/CloudCompare step4.png" width="40%" height="auto"/>
</p>

**Robot Base Placement** Position the robot's base on tape1 (pass its coordinates as x1 and y1 when running codes on robot side) and orient it towards tape2 (pass its coordinates as x2 and y2 when running codes on robot side) as shown in below images. Make sure the position and rotation of the robot are as accurate as possible as it is crucial for the experiments to run properly. We encourage you to manually finetune the robot's positions or tapes' coordinates if you find the tape coordinates are not accurate enough (You can try to run one navigation query check navigation path planning visualization to see the robot's target position, if the robot is obviously off from the target position in visualization, then you are encouraged to tune robot's initial positions)

<p align="center">
  <img src="docs_image/How you should use tape to localize robot1.png" width="25%" height="380" style="margin-right: 10px;"/>
  <img src="docs_image/How you should use tape to localize robot2.png" width="25%" height="380"/>
</p>

