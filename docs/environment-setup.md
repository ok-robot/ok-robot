# Experiment Setup
**Environment Setup:** Use Record3D to scan the environments. Recording should include: 
* All Obstacles in environments
* Complete floor where the robots can navigate
* All testing objects.
* Two tapes in the scene which will serve as a orgin for the robot.

**Tape Placement:** This [drive folder](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) has illustrations on how to place tapes on the ground and scan the environment properly.

**Scan the Environment:** After positioning the objects and tapes, proceed to scan the environment and save the Record3D r3d file in the `ok-robot-navigation/r3d/` folder. If you just want to try out a sample r3d file, you can use `ok-robot-navigation/r3d/sample.r3d`.

**Extract PointCloud:** Then, use the `ok-robot-navigation/get_point_cloud.py` script to extract the pointcloud of the scene, which will be stored as `ok-robot-navigation/pointcloud.ply`. 
```
python get_point_cloud.py --input_file=[your r3d file]
```
**Identify Tape Co-ordinates** Use a 3D Visualizer to determine the coordinates of the two orange tapes in the environment. Let the co-ordinates of these tapes are (x1, y1) for the first tape(tape1) and (x2, y2) for the second tape(tape2).

**Robot Base Placement** Position the robot's base on tape1 and orient it towards tape2 as shown in the [image](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw). Make sure the position and rotation of the robot as accurate as possible as it is crucial for the experiments to run properly. 

We recommend using CloudCompare to localize coordinates of tapes. See the [google drive folder above](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) to see how to use CloudCompare.

