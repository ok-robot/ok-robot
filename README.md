## Hardware and software requirements
Hardware required:
* iPhone with Lidar sensors
* Stretch re1 robot
* A workstation machine to run pretrained models 
  
Software required:
* Python 3.9
* [CloudCompare](https://www.danielgm.net/cc/release/) (a pointcloud processing software)
* Record3D (installed on iPhone) [Has to mention the version]
* Other software packages needed for running pretrained models (e.g. Python)

## Clone this repository to your local machine
```
git clone https://github.com/NYU-robot-learning/home-engine
git submodule update --init --recursive
```

## Workspace Installation and setup
install Mamba if it is not present 
```
mamba env create -n ok-robot-env -f ./environment.yml
mamba activate ok-robot-env

pip install graspnetAPI

# Setup Home-robot
cd home-robot
python -m pip install -e src/home_robot
cd ..

# Setup poincept
cd anygrasp/pointnet2/
python setup.py install
```

## Anygrasp Setup
Get license from so and so

```
cd ./anygrasp
cp gsnet_versions/gsnet.cpython-39-x86_64-linux-gnu.so src/gsnet.so
cp license_registration/lib_cxx_versions/lib_cxx.cpython-39-x86_64-linux-gnu.so
```

Place the license folder in anygrasp/src and checkpoint.tar in anygrasp/src/checkpoints/

## Installation Verification
Verify whether you are able to succesfully run path_planning.py file. It should run succesfully and you see a prompt asking to enter A
```
python path_planning.py debug=True
```

Then verify whether the grasping module is running properly. It should ask prompts for task [pick/place] and object of interest. You can view in scene image in /anygrasp/src/example_data/peiqi_test_rgb21.png. Choose a object in the scene and you see visualizations showing a grasp around the object and green disk showing the area it want to place.
```
cd anygrasp/src
./demo.sh
```

## Robot Installation and setup
Enter home-robot submodule
```
cd $OK-Robot/home-robot
```

Follow the [home-robot installation instructions](https://github.com/leo20021210/home-robot/blob/main/docs/install_robot.md) to install home-robot on your Stretch robot.

Also follow [home-robot calibration instructions](https://github.com/leo20021210/home-robot/blob/main/docs/calibration.md) to calibrate your robot.

After that run the following commands to move calibrated urdf to robot controller
```
cd $OK-Robot/
cp home-robot/assets/hab_stretch/ GrasperNet
```

## Environment Setup
Use Record3D to scan the environments. Recording should include: 
* All Obstacles in environments
* Complete floor where the robots can navigate
* All testing objects.
* Two tapes in the scene which will serve as a orgin for the robot.

Take a look at this [drive folder](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) and gain insights on how you should place tapes on the ground, how you should scan the environment.

After you obstain a .r3d file from Record3D. If you just want to try out a sample, download the [sample data](https://osf.io/famgv) `nyu.r3d` and run the following command. Place it in the `/voxel_map/r3d/` folder. 


Use the `get_point_cloud.py` script to extract the pointcloud of the scene. This will store the point cloud `pointcloud.ply` in home directory. 
```
python get_point_cloud.py --input_file=[your r3d file]
```
Extract the point co-ordinates of two orange tapes using a 3D Visualizer. Let (x1, y1), (x2, y2) be the co-ordiantes of two tapes [tape1, tape2] then place the robot base on tape1 and orient it towards tape2 like this[will attach a image]

We recommend using CloudCompare to localize coordinates of tapes. See the [google drive folder above](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) to see how to use CloudCompare.

## Load voxel map 
Once you setup the environment run voxel map with your r3d file which will save the semantic information of 3D Scene in `/voxel-map` directory.
```
cd voxel-map
python load_voxel_map.py dataset_path=[your r3d file location]
```
You can check other config settings in `/voxel-map/configs/train.yaml`.

To modify these config settings, you can either do that in command line or by modifying YAML file.

After this process finishes, you can check your stored semantic map in the path specified by `cache_path` in `/voxel-map/configs/train.yaml`

## Config files
### `/voxel-map/config/train.yaml`
It contains parameters realted to training the voxel map. Some of the important parameters are
* **dataset_path** - path to your r3d file
* **cache_path** - path to your semantic information file
* **sample_freq** - sampling frequency of frames while training voxel map
* **threshold** - threshold for your segmentation models such as OWL-ViT and Detic
* **subsample_prob** - sampling probability for points extracted from each R3D video frame.
* **custom_labels** - some additional labels you want to add for OWL-ViT to detect.

### `/path.yaml`
Contains parameters related to path planning
* **min_height** - z co-ordinate value below which everything is considered as non-navigable points. Ideally you should choose a point on the floor of the scene and set this value slightly more than that [+0.5 or 1].
* **max_height** - z co-ordiante of the scene above which everything is neglected.
* **map_type** - conservative_vlmap or brave_vlmap Fill more abt this @peiqi
* **localize_type** - 
* **resolution** - 
* **occ_avoid_radius** - 

### Running experiments
Once the above config filesa reset you can start running experiments

Scan the environments with Record3D, follow steps mentioned above in Environment Setup and Load Voxel map to load semantic map and obstacle map. Move the robot to the starting point specified by the tapes or other labels marked on the ground.

Start home-robot on the Stretch:
```
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

Navigation planning (you do not specify anything other than fields in `/path.yaml` as it will automatically read the fields in your voxel map config file such as `/voxel-map/configs/train.yaml`):
```
python path_planning.py

```

Pose estimation:
```
bash demo.sh
```

Robot control:
```
python run.py -x1 [x1] -y1 [y1] -x2 [x2] -y2 [y2]
```

