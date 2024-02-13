# Installation Verification
### Navigation module Testing

Run `path_planning.py`. It should ask for Object Name [A:] and a near by object [B:]. Then you should see 3D visualisations of localized object point and and a 2D Visualization saved in `navigation/test/{object_name}`. More about this in navigation [README](./navigation/README.md)
```
cd navigation

python path_planning.py debug=True
cd ../
```

### Manipulation Module Testing
Run demo.py. It should ask for action [pick/place] and object name from `/anygrasp/src/example_data/ptest_rgb.png` file. Then you should see visualisation related to all the predicted grasps in the scene and a final grasp related to object. More about this in manipulation [README](./manipulation/README.md). [If you face any memory issues try reducing the sampling rate to 0.2 in `anygrasp/src/demo.py`]
```
cd anygrasp/src

python demo.py --debug 
# run in debug mode to see 3D Manipulation Visualisations. If you are running remotely its better to avoid this option
```

If you are able to see the visualizations then workstation setup is complete. You can move onto setting up on robot.
