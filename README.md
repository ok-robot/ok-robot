
![Intro Figure](https://drive.google.com/uc?export=view&id=1IAyAMZS__gcZmsZevQyeETLU369a0n9X)
# Ok-Robot

[<u>Project Website</u>](https://ok-robot.github.io/) . [<u>Paper</u>](https://arxiv.org/abs/2401.12202)

**Authors List**: [<u>Peiqi Liu</u>](https://leo20021210.github.io/), [<u>Yaswanth Orru</u>](https://www.linkedin.com/in/yaswanth-orru/), [<u>Jay Vakil</u>](https://www.linkedin.com/in/jdvakil/), [<u>Chris Paxton</u>](https://cpaxton.github.io/), [<u>Mahi Shafiuallah</u>](https://mahis.life/), [<u>Lerrel Pinto</u>](https://www.lerrelpinto.com/) 

Ok-Robot is a framework that combines the state-of-art navigation and manipualtion models in a intelligent way to design a modular system that can effectively perform pick and place tasks in real homes. It has been tested in 10 real homes on 170+ objects and achieved a total success rate of 58%. 

<details open>
  <summary>ok-robot.mp4</summary>
  

  <!-- Your embedded video code goes here -->
  <video src="https://drive.google.com/file/d/1IC_wUHmnvq_Aon-fKdiNUNRawQC3patP/view?usp=sharing" width="640" height="480"></video>
</details>

## Hardware and software requirements
Hardware required:
* iPhone with Lidar sensors
* Stretch robot with Dex Wrist installed
* A workstation machine to run pretrained models 
  
Software required:
* Python 3.9
* Record3D (>1.18.0)

## Installation
* First you need to get the [license and checkpoint](./anygrasp/license_registration/README.md) for anygrasp.
* [Install](./docs/workspace-installation.md) the necessary environment on workstation to run the navigation and manipulation modules
* [Install](./docs/robot-installation.md) the necessary packages on robot to abe able to properly communicate with backend workstation.
* You might also need to get a [new calibrated URDF](./docs/robot-calibration.md) for accurate robot manipulation.
* Once the above things are done [verify the installation](./docs/installation-verification.md).

Once the robot and workstation are complete. You are good to start the experiments.

## Run Experiments
First [set up the environment](./docs/environment-setup) with the tapes and position the robot properly and scan the environment and get a r3d file from Record3D and place it in `/navigation/r3d/` run following commands.

**On Workstation**:

In one terminal run the [Navigation Module](./navigation/).
```
mamba activate ok-robot-env

cd navigation
python path_planning.py debug=False dataset_path='/r3d/{your_r3d_filename}.r3d' cache_path={your_r3d_filename}.pt
```

In another terminal run the [Manipulation module](./anygrasp/README.md)
```
mamba activate ok-robot-env

cd anygrasp
python dempy.py --open_communication --debug
```

**On Robot**:

In one terminal start the home-robot
```
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

In another terminal run the robot control. More details in [graspernet](./graspernet/README.md)
```
python run.py -x1 [x1] -y1 [y1] -x2 [x2] -y2 [y2] -ip [your workstation ip]

```