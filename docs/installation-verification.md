# Installation Verification
### Navigation module Testing

Run `path_planning.py`. It should ask for Object Name [A:] and a near by object [B:]. Then you should see 3D visualisations of localized object point and and a 2D Visualization saved in `ok-robot-navigation/test/{object_name}`. More about this in navigation [README](../ok-robot-navigation/README.md)
```
cd ok-robot-navigation

python path_planning.py debug=True
# run in debug mode to see 3D Manipulation Visualisations. If you are running in a headless mode (e.g. remotely, or on a machine not connected with monitors) its better to avoid this option. 
# In the headless mode, the command you should run would be python path_planning.py debug=True pointcloud_visualization=False
cd ../
```

### Manipulation Module Testing
Run demo.py. It should ask for action [pick/place] and object name from `/ok-robot-manipulation/src/example_data/ptest_rgb.png` file. Then you should see visualisation related to all the predicted grasps in the scene and a final grasp related to object. More about this in manipulation [README](../ok-robot-manipulation/README.md). [If you face any memory issues try reducing the sampling rate to 0.2 in `ok-robot-manipulation/src/demo.py`]
```
cd ok-robot-manipulation/src

python demo.py --debug 
# run in debug mode to see 3D Manipulation Visualisations. If you are running in a headless mode (e.g. remotely, or on a machine not connected with monitors) its better to avoid this option. 
# In the headless mode, the command you should run would be python demo.py
```

If you are able to see the visualizations then workstation setup is complete. You can move onto setting up on robot.
