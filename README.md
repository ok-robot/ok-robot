# Home_engine
## Installation
To install navigation, run these scripts
```
mamba create -n home_engine python=3.10
mamba activate home_engine
mamba install -y -c "nvidia/label/cuda-11.8.0" -c pytorch -c nvidia cuda-toolkit pytorch torchvision torchaudio
cd ..
python -m pip install -r requirements.txt
cd clip-fields/gridencoder/
python setup.py install

# install required USA-Net packages
cd ../..
cd usa
pip install -e .

# install required home-robot packages
cd ..
cd home-robot/src/home-robot
pip install -e .
cd ../../..
```

You should also follow these [instructions](https://github.com/NYU-robot-learning/anygrasp/blob/4c7a9d465a85591a2a4d99c5eb709018ea26b0a6/grasp_detection/README.md) to install a conda environment for AnyGrasp.

## Hardware and software requirements
Hardware required:
* iPhone with Lidar sensors
* Stretch robot
* A workstation machine that can use to run pretrained models
  
Software required:
* [CloudCompare](https://www.danielgm.net/cc/release/) (a pointcloud processing software)
* Record3D (installed on iPhone)
* Other software packages needed for running pretrained models (e.g. Python)
  
## Running experiments
After setting up environments and putting testing objects in the environments, you can start running experiments. Here shows an example environment
### Scan the environments
To align the robot coordinate system (the one robot uses to localizes itself) and navigation coordinate system (the one provided by Record3D and used by navigation stack), we generally put two tapes on the ground.

Use Record3D to scan the environments. Recording should include: 
* all obstacles in environments
* the floor where the robots can navigate onto
* all testing objects.

Take a look at this [drive folder](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) and gain insights on how you should place tapes on the ground, how you should scan the environment.

After you obstain a .r3d file from Record3D, you should localize the coordinates of two tapes and save it in a notepad for using them in later steps.

We recommend using CloudCompare to localize coordinates of tapes. See the [google drive folder above](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing) to see how to use CloudCompare.
### Load navigation stack
#### "Train" voxel map
Once you have the dependencies installed, you can run the training script `train.py` with any [.r3d](https://record3d.app/) files that you have! If you just want to try out a sample, download the [sample data](https://osf.io/famgv) `nyu.r3d` and run the following command.

```
cd clip-fields
python train.py dataset_path=nyu.r3d
```
You can check out the `config/train.yaml` for a list of possible configuration options. In particular, if you want to train with any particular set of labels, you can specify them in the `custom_labels` field in `config/train.yaml`.
#### Update other necessary config files
You should also edit those config files:
* In `usa/configs/train.yaml`, modify field `task/dataset_path` to .r3d file you used for "training" voxel map.
* In `path.yaml` you should modify `min_height` and `max_height` fields as they are floor heights and ceil heights we used for loading navigation obstacle map. Generally you should set `min_height` slightly higher (10cm) than z coordinates of orange tapes.
* Next you might run `python path_planning.py` to start navigation planning.
### Load manipulation stack
* In `anygrasp` folder you should run `bash demo.sh` to start grasping pose estimation
* You should follow [home-robot instructions](https://github.com/leo20021210/home-robot) to install home-robot packages either on workstation or on robots.
* Place your robot following [google drive folder above](https://drive.google.com/drive/folders/1qbY5OJDktrD27bDZpar9xECoh-gsP-Rw?usp=sharing).
* You should run robot controller in `GrasperNet` folder by run `python run.py -bf top_camera -t -x_offset [x_offset] -y_offset [y_offset] -theta_offset [theta_offset]`.
* When running the experiments, three processes should run simultaneously, `python path_planning.py` for navigation path planning, `bash demo.sh` for pose estimation, and `python run.py` for robot controlling
