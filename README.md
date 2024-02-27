[![Intro image](https://github.com/ok-robot/ok-robot/assets/156981262/5634372e-9820-4681-bb1a-1c14541c229c)](https://ok-robot.github.io)


# `OK-Robot`

[![arXiv](https://img.shields.io/badge/arXiv-2401.12202-163144.svg?style=for-the-badge)](https://arxiv.org/abs/2401.12202)
![License](https://img.shields.io/github/license/notmahi/bet?color=873a7e&style=for-the-badge)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-262626?style=for-the-badge)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/Videos-Website-db6a4b.svg?style=for-the-badge&logo=airplayvideo)](https://ok-robot.github.io/)
[![Discord](https://img.shields.io/discord/1210368035324436551?style=for-the-badge&logo=discord&color=5865F2&label=discord&logoColor=white)](https://discord.gg/wzzZJxqKYC)

**Authors**: [<u>Peiqi Liu</u>*](https://leo20021210.github.io/), [<u>Yaswanth Orru</u>*](https://www.linkedin.com/in/yaswanth-orru/), [<u>Jay Vakil</u>](https://www.linkedin.com/in/jdvakil/), [<u>Chris Paxton</u>](https://cpaxton.github.io/), [<u>Mahi Shafiuallah</u>](https://mahis.life/)<sup>†</sup>, [<u>Lerrel Pinto</u>](https://www.lerrelpinto.com/)<sup>†</sup>    
\* equal contribution, † equal advising.

OK-Robot is a zero-shot modular framework that effectively combines the state-of-art navigation and manipulation models to perform pick and place tasks in real homes. It has been tested in 10 real homes on 170+ objects and achieved a total success rate of 58.5%. 

https://github.com/ok-robot/ok-robot/assets/156981262/7221264b-2fa3-4008-a757-0604cdc04e2b

## Hardware and software requirements
Hardware required:
* An iPhone Pro with Lidar sensors
* [Hello Robot Stretch](https://hello-robot.com/) with Dex Wrist installed
* A workstation with GPU to run pretrained models 

Software required:
* Python 3.9
* Record3D (>1.18.0)
* [CloudCompare](https://www.danielgm.net/cc/release/)

## Installation
* You need to get anygrasp [license and checkpoint](./ok-robot-manipulation/anygrasp_license_registration/README.md).
* [Install](./docs/workspace-installation.md) the necessary environment on workstation to run the navigation and manipulation modules
* [Verify the workspace installation](./docs/installation-verification.md) once the above steps are completed.
* [Install](./docs/robot-installation.md) the necessary packages on robot to be able to properly communicate with backend workstation.
* You might also need to get a [new calibrated URDF](./docs/robot-calibration.md) for accurate robot manipulation.

Once both the robot and workstation are complete. You are good to start the experiments.

## Run Experiments

First [set up the environment](./docs/environment-setup.md) with the tapes, position the robot properly and scan the environment to get a r3d file from Record3D. Place it in `/navigation/r3d/` run following commands.


### On Workstation:

In one terminal run the [Navigation Module](./ok-robot-navigation/).
```
mamba activate ok-robot-env

cd ok-robot-navigation
python path_planning.py debug=False min_height={z coordinates of the ground tapes + 0.1} dataset_path='r3d/{your_r3d_filename}.r3d' cache_path='{your_r3d_filename}.pt' pointcloud_path='{your_r3d_filename}.ply'
```

In another terminal run the [Manipulation module](./ok-robot-manipulation/README.md)
```
mamba activate ok-robot-env

cd ok-robot-manipulation/src
python demo.py --open_communication --debug
```

### On Robot:

Before running anything on the robot, you need to calibrate it by 
```
stretch_robot_home.py
```

Our robot codes rely on robot controllers provided by [home-robot](https://github.com/facebookresearch/home-robot). Just like running other home-robot based codes, you need to run two processes synchronously in two terminals.

In one terminal start the home-robot
```
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

In another terminal run the robot control. More details in [ok-robot-hw](./ok-robot-hw/README.md)
```
cd ok-robot-hw

python run.py -x1 [x1] -y1 [y1] -x2 [x2] -y2 [y2] -ip [your workstation ip]
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{liu2024okrobot,
  title={OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics},
  author={Liu, Peiqi and Orru, Yaswanth and Paxton, Chris and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2401.12202},
  year={2024}
}
```

Our work is reliant on a lot of other publications and open source projects, if you find a particular component useful, please consider citing the original authors as well.

<details>
<summary>List of citations</summary>

```bibtex
@article{fang2023anygrasp,
  title={Anygrasp: Robust and efficient grasp perception in spatial and temporal domains},
  author={Fang, Hao-Shu and Wang, Chenxi and Fang, Hongjie and Gou, Minghao and Liu, Jirong and Yan, Hengxu and Liu, Wenhai and Xie, Yichen and Lu, Cewu},
  journal={IEEE Transactions on Robotics},
  year={2023},
  publisher={IEEE}
}

@article{minderer2024scaling,
  title={Scaling open-vocabulary object detection},
  author={Minderer, Matthias and Gritsenko, Alexey and Houlsby, Neil},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@article{yenamandra2023homerobot,
  title={HomeRobot: Open-Vocabulary Mobile Manipulation},
  author={Yenamandra, Sriram and Ramachandran, Arun and Yadav, Karmesh and Wang, Austin and Khanna, Mukul and Gervet, Theophile and Yang, Tsung-Yen and Jain, Vidhi and Clegg, Alexander William and Turner, John and others},
  journal={arXiv preprint arXiv:2306.11565},
  year={2023}
}
```
</details>

## Roadmap

While OK-Robot can do quite a bit by itself, we think there are plenty of room for improvement for a zero-shot, home-dwelling robot. That's why we consider OK-Robot a living release, and will try to occassionally add new features to this. We also encourage you to take a look at the list below, and if you are interested, share your improvements with the community by contributing to this project.

- [x] Create OK-Robot, a shared platform for a zero-shot, open-vocab pick-and-place robot.
- [x] Integrate grasping primitive with AnyGrasp.
- [x] Integrate open-vocabulary navigation with VoxelMap.
- [x] Integrate heuristic based dropping.
- [ ] Improve documentation.
- [ ] Add error detection/recovery from failure while manipulating.
- [ ] Figure out interactive navigation: if an object is not found or a query is ambiguous, ask the end-user.
- [ ] Integrate with an open-source grasp perception model so that we can MIT-license all the dependencies.
