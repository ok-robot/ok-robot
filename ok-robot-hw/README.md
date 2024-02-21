# Robot Side Code
Most of the heavy code will be runnning in the workstation and will communicate with the robot through sockets

## Preparation to run robot side codes
Our hardware codes heavily rely on robot controller provided by [home-robot](https://github.com/facebookresearch/home-robot).

You need to install the home-robot on stretch robot following intructions provided by [home-robot-hw](https://github.com/facebookresearch/home-robot/blob/main/docs/install_robot.md) before running any robot side codes on robot.

To check whether home-robot is installed properly and got familiar with running home-robot based codes, we recommend you try to run [these test scripts](https://github.com/facebookresearch/home-robot/blob/main/tests/hw_manual_test.py).

Once you finised installing home-robot, follow [these steps](../docs/robot-installation.md) to enable OK-Robot use home-robot contollers.

The success of OK-Robot system also depends on robot calibration and accurate urdf file, so make sure you follow [these calibration steps](../docs/robot-calibration.md) to obtain an accurate urdf file for your robot.

## Start home-robot
```
stretch_robot_home.py
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

## Start Robot Control
```
python run.py -x1 [x1] -y1 [y1] -x2 [x2] -y2 [y2] -ip [your workstation ip]
```

* **[x1, y1]** - Co-ordinated of tape on which the base of the robot is on
* **[x2, y2]** - Co-ordinates of the secondary tape.
* **ip** - Your workstation ip, the robot will try to communicate with this ip
* **np** - Navigation port number, the robot will listen to this port number to get planned navigation paths (default: 5555)
* **mp** - Manipulation port number, the robot will listen to this port number to get estimated manipulation pose (default: 5556)

After running run.py it will go through 4 states in each cycle. 
* Picking Navigation
* Manipulation
* Placing Navigation
* Placing 

For each navigation stage it asks A [Object Name], B [Near by Object Name]