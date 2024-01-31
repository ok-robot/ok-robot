# Home_engine
## Issues
- KeyError jointwrist pitch
- * grdiencoder "CUDA_HOME=/usr/local/cuda-11.7"
- No such file or directory: 'clip-fields/Yaswanth_Bedroom_model_weights/implicit_scene_label_model_latest.pt
- AssertionError: Torch not compiled with CUDA enabled
- assert len(conf_fnames) == tsz [Record 3d issue]
- No reachable points [Check min-height, ]

## Hardware and software requirements
Hardware required:
* iPhone with Lidar sensors
* Stretch robot
* A workstation machine that can use to run pretrained models
  
Software required:
* Python 3.9
* [CloudCompare](https://www.danielgm.net/cc/release/) (a pointcloud processing software)
* Record3D (installed on iPhone) [Has to mention the version]
* Other software packages needed for running pretrained models (e.g. Python)

## Workspace Installation and setup
install Mamba if it is not present 
```
mamba env create -n ok-robot-env -f ./environment1.yml
mamba activate ok-robot-env

pip install graspnetAPI

# Setup Home-robot
cd home-robot
python -m pip install -e src/home_robot
cd ..

# Setup poincept
cd anygrasp/pointnet2/
python setup.py install
cd ../../
```

See anygrasp/README.MD for additional setup required for grasping module [Will be updated once the anygrasp directory is set]

## Installation Verification
Verify whether you are able to succesfully run path_planning.py file. It should run succesfully and you see a prompt asking to enter A
'''
python path_planning.py
'''

Then verify whether the grasping module is running properly. It should ask prompts for task [pick/place] and object of interest. You can view in scene image in /anygrasp/src/example_data/peiqi_test_rgb21.png. Choose a object in the scene and you see visualizations showing a grasp around the object and green disk showing the area it want to place.
'''
cd anygrasp/src
./demo.sh
'''

## Running experiments
After setting up environments and putting testing objects in the environments, you can start running experiments.

### Scan the environments
To align the robot coordinate system (the one robot uses to localizes itself) and navigation coordinate system (the one provided by Record3D and used by navigation stack), we generally put two tapes on the ground.

Use Record3D to scan the environments. Recording should include: 
* all obstacles in environments
* the floor where the robots can navigate onto
* all testing objects.

Take a look at this [drive folder](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) and gain insights on how you should place tapes on the ground, how you should scan the environment.

After you obstain a .r3d file from Record3D, you should localize the coordinates of two tapes and save it in a notepad for using them in later steps.

Use the [get_point_cloud](utils/get_point_cloud.py) script to extract this PLY file.
Extract `pointcloud.ply` pointcloud from .r3d file with following python script (after running scripts, you will have a ply file named `pointcloud.ply` in your folder that represents this environment).

After this, you can load and visualize the point cloud.

We recommend using CloudCompare to localize coordinates of tapes. See the [google drive folder above](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) to see how to use CloudCompare.
### Load navigation stack
#### "Train" voxel map
Once you have the dependencies installed, you can run the training script `train.py` with any [.r3d](https://record3d.app/) files that you have! If you just want to try out a sample, download the [sample data](https://osf.io/famgv) `nyu.r3d` and run the following command.

```
cd clip-fields
python train.py dataset_path=nyu.r3d
```
You can check out the `config/train.yaml` for a list of possible configuration options. In particular, if you want to train with any particular set of labels, you can specify them in the `custom_labels` field in `config/train.yaml`.

Change the location in `clip-fields/config.train.yaml`; for example:
```
cache_path: ChrisHome.pt
saved_dataset_path: ChrisHome.pt
```

This is where data will be stored.

#### Update other necessary config files
You should also edit those config files:
* In `usa/configs/train.yaml`, modify field `task/dataset_path` to .r3d file you used for "training" voxel map.
* In `path.yaml` you should modify `min_height` and `max_height` fields as they are floor heights and ceil heights we used for loading navigation obstacle map. Generally you should set `min_height` slightly higher (10cm) than z coordinates of orange tapes.
* Next you might run `python path_planning.py` to start navigation planning.

### Load manipulation stack
* In `anygrasp` folder you should run `bash demo.sh` to start grasping pose estimation
* You should follow [home-robot instructions](https://github.com/leo20021210/home-robot) to install home-robot packages either on workstation or on robots.
* Place your robot following [google drive folder above](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing).
* Given the coordinates of tape robot stands on `(x1, y1)` and coordinates of tape robot faces to `(x2, y2)` you should run robot controller in `GrasperNet` folder by run `python run.py -bf top_camera -t -x1 [x1] -y1 [y1] -x2 [x2] -y2 [y2]`.
* When running the experiments, three processes should run simultaneously, `python path_planning.py` for navigation path planning, `bash demo.sh` for pose estimation, and `python run.py` for robot controlling

### Running experiments

Start home-robot on the Stretch:
```
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

Navigation planning:
```
python path_planning.py

```

Pose estimation:
```
bash demo.sh
```

Robot control:
```
python run.py
```

