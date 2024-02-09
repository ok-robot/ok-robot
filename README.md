# Home_engine
<!-- ## Previous encountered setup Issues [just to keep track will be removed afterwards] -->
<!-- - KeyError jointwrist pitch [Removed inn latest upgrades]
- grdiencoder "CUDA_HOME=/usr/local/cuda-11.7" []
- No such file or directory: 'clip-fields/Yaswanth_Bedroom_model_weights/implicit_scene_label_model_latest.pt [Have to be document properly]
- AssertionError: Torch not compiled with CUDA enabled [torch installation. Removed in latest build]
- assert len(conf_fnames) == tsz [Record 3d issue]
- No reachable points [Check min-height, ] -->

## Hardware and software requirements
Hardware required:
* iPhone with Lidar sensors
* Stretch robot with Dex Wrist installed
* A workstation machine to run pretrained models 
  
Software required:
* Python 3.9
* [CloudCompare](https://www.danielgm.net/cc/release/) (a pointcloud processing software)
* Record3D (installed on iPhone) [has to mention the version]
* Other software packages needed for installing python packages (e.g. Mamba, conda)

## Clone this repository to your local machine
```
git clone https://github.com/NYU-robot-learning/home-engine
cd home-engine
git checkout peiqi
git submodule update --init --recursive
```

## Workspace Installation and setup
install Mamba if it is not present 

### CUDA 12.*
```
# Environment creation
mamba env create -n ok-robot-env -f ./env-cu121.yml
mamba activate ok-robot-env

# Pointnet setup for anygrasp
cd anygrasp/pointnet2/
python setup.py install
cd ../../

# Additional pip packages isntallation
pip install -r requirements-cu121.txt
pip install --upgrade --no-deps --force-reinstall scikit-learn==1.4.0 (any issue realted to sklearn)
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html (if you are not able to import torch cluster properly)
pip install graspnetAPI

```

### CUDA-11.8
```
mamba env create -n ok-robot-env -f ./env-cu118.yml
mamba activate ok-robot-env

pip install --upgrade --no-deps --force-reinstall scikit-learn==1.4.0
pip install graspnetAPI

# Setup poincept
cd anygrasp/pointnet2/
python setup.py install
```

## Anygrasp Setup
Please refer licence [readme](/anygrasp/license_registration/README.md) for more details on how to get the anygrasp license. Once this process is done you will receive the license and checkpoint.tar through email.
Place the license folder in anygrasp/src and checkpoint.tar in anygrasp/src/checkpoints/

## Installation Verification
<!-- Load Voxel Map. This is should create a sample.pt file in `/Navigation/voxel-map/` folder
```
cd navigation/voxel-map/
python load_voxel_map.py
cd ../
``` -->

Run `/Navigation/path_planning.py` file and set debug be True. If run succesfully, the process will load the semantic memory from record3D file, and you should see a prompt asking to enter A. Enter a object name and you should see a 2d map of the scene with the object localised with a green dot in `Navigation/test/{object_name}` folder. 
```
python path_planning.py debug=True
cd ../
```

Then verify whether the grasping module is running properly. It should ask prompts for task [pick/place] and object of interest. You can view in scene image in /anygrasp/src/example_data/peiqi_test_rgb21.png. Choose a object in the scene and you see visualizations showing a grasp around the object and green disk showing the area it want to place. [If you facing any memory issues try reducing the sampling rate to 0.2 in `anygrasp/src/demo.py`]
```
cd anygrasp/src
python demo.py --checkpoint_path checkpoints/checkpoint_detection.tar --debug
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

## Testing Environment Setup
Use Record3D to scan the environments. Recording should include: 
* All Obstacles in environments
* Complete floor where the robots can navigate
* All testing objects.
* Two tapes in the scene which will serve as a orgin for the robot.

Take a look at this [drive folder](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) and gain insights on how you should place tapes on the ground, how you should scan the environment.

After you obstain a .r3d file from Record3D place it in the `navigation/r3d/` folder. If you just want to try out a sample, you can use `navigation/r3d/sample.r3d`.


Use the `navigation/get_point_cloud.py` script to extract the pointcloud of the scene. This will store the point cloud `navigation/pointcloud.ply` of the scene. 
```
python get_point_cloud.py --input_file=[your r3d file]
```
Extract the point co-ordinates of two orange tapes using a 3D Visualizer. Let (x1, y1), (x2, y2) be the co-ordiantes of two tapes [tape1, tape2] then place the robot base on tape1 and orient it towards tape2 like this[will attach a image]

We recommend using CloudCompare to localize coordinates of tapes. See the [google drive folder above](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) to see how to use CloudCompare.

<!--## Load voxel map 
Once you setup the environment run voxel map with your r3d file which will save the semantic information of 3D Scene in `Navigation/voxel-map` directory. 
```
cd navigation/voxel-map
python load_voxel_map.py dataset_path=[your r3d file location]
```
You can check other config settings in `navigation/voxel-map/configs/train.yaml`.

To modify these config settings, you can either do that in command line or by modifying YAML file.

After this process finishes, you can check your stored semantic map in the path specified by `cache_path` in `Navigation/voxel-map/configs/train.yaml`-->

## Config files `Navigation/configs/path.yaml` 40a17efae7bc154518cc8b8830b25fe63776146c
It contains parameters realted to training the voxel map. Some of the important parameters are
* **debug** - debug=True: generate localization result for each text query; debug=False: communicate with robots and send planned path to robots
* **dataset_path** - path to your r3d file
* **cache_path** - path to your semantic information file
* **sample_freq** - sampling frequency of frames while training voxel map
* **min_height** - z co-ordinate value below which everything is not considered as obstacles. Ideally you should choose a point on the floor of the scene and set this value slightly more than that [+ 0.1].
* **max_height** - z co-ordiante of the scene above which everything is not considered as obstacles.
* **map_type** - "conservative_vlmap" (space not scanned is non-navigable) "brave_vlmap" (sapce scanned is navigable, used when you forget to scan the floor)
* **port_number** - zmq communication port, need to the the same with robot's navigation port number
* **save_file** - planned path and generated localization results will be saved here 

## Running experiments
Once the above config files are set you can start running experiments

Scan the environments with Record3D, follow steps mentioned above in Environment Setup and Load Voxel map to load semantic map and obstacle map. Move the robot to the starting point specified by the tapes or other labels marked on the ground.

Start home-robot on the Stretch:
```
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

Navigation planning (See `Navigation/configs/path.yaml` for any config settings):
```
cd navigation
python path_planning.py debug=False
```

Pose estimation:
```
cd anygrasp/src/
bash demo.sh
```

Robot control:
```
python run.py -x1 [x1] -y1 [y1] -x2 [x2] -y2 [y2]
```

