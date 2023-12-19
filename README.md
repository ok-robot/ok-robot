# Home_engine
## Installation
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
<a href="https://drive.google.com/uc?export=view&id=<FILEID>"><img src="https://drive.google.com/uc?export=view&id=1O-y5vhRuSZfgJ_ENjg-bTAHXOLvGtKGo" style="width: 650px; max-width: 100%; height: auto" title="Click to enlarge picture" />
### Scan the environments
To align the robot coordinate system (the one robot uses to localizes itself) and navigation coordinate system (the one provided by Record3D and used by navigation stack), we generally put two tapes on the ground.\
Use Record3D to scan the environments. Recording should include: 1. all obstacles in environments 2. the floor where the robots can navigate onto 3. all testing objects.
### Load navigation stack
### Load manipulation stack
