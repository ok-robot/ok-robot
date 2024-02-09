# Robot Side Code
Most of the heavy code will be runnning in the workstation and will communicate with the robot through sockets

Once you have installed the home-robot on stretch robot run the following codes on robot 
## Start home-robot
```
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

## Start Robot Control
```
python run.py -x1 [x1] -y1 [y1] -x2 [x2] -y2 [y2]
```

* **[x1, y1]** - Co-ordinated of tape on which the base of the robot is on
* **[x2, y2]** - Co-ordinates of the secondary tape.

After running run.py it will go through 4 states in each cycle. 
* Picking Navigation
* Manipulation
* Placing Navigation
* Placing 

For each navigation stage it asks A [Object Name], B [Near by Object Name]