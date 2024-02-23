<!-- <img src="https://user-images.githubusercontent.com/12446953/208367719-4ef7922f-4001-41f7-aa9f-076e462d1325.png" width="60%"> -->

# Manipulation Module
This section contains Manipulation related code. Below figure describes high level pipeline of how a final grasp is generated from a RGBD Image taken from robot head camera.
* [Anygrasp](https://arxiv.org/abs/2212.08333) generates all possible poses in the scence
* [Lang-sam](https://github.com/luca-medeiros/lang-segment-anything) is used to extract the object mask
* The grasps points are projected back onto the image and filtered to retain only ones that fall on the mask.
* The grasp with max score is chosen as final grasp.

![Manipulation Pipeline](https://drive.google.com/uc?export=view&id=1H7ddROUyjqFDhEMOOyr_-vcOqFwm203o)

## License Registration
There isn't any open-source code available for anygrasp at the moment. Therefore, you'll need to obtain an anygrasp SDK. To do that, retrieve the feature ID of your machine and fill the [form](https://forms.gle/XVV3Eip8njTYJEBo6) to apply for the license.

## Code
Once you have installed the environment following [these](../README.md#workspace-installation-and-setup) instructions. Run the following code
```
mamba activate ok-robot-env

cd /src/
python demo.py --debug
```
**debug flag -** Is for visualizing 3d plots of grasping.
**open_communication flag -** Is for selection from dry run and robot running (here we only want to dry run to test environment installation).

**open_communication -** When you are operating with robot

## Visualizations
Once you run the program it will save the following visualizations in the `save_directory` option in `src/demo.py` file
* **clean_*.jpg -** Image taken by the robot for processing
* **semantic_segmentation_*.jpg -** Segemented object mask of the query
* **poses.jpg -** Screen shot of 3d scene with all the predicted poses
* **grasp_projections.jpg -** Green dots indicate grasps inside the object mask and red dots indicate grasps outside the object mask.
* **best pose.jpg -** Final executed pose

These visualizations help in understanding the output behaviour and also helps in debugging in case of errors.

## Debugging

### GLFW Error
If you encounter an error that looks something like this:
```
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
[Open3D WARNING] GLFW Error: ...
```

You are probably trying to run the code on a remote server. Unfortunately, the support for headless rendering for Open3D isn't great, and the easiest solution is to follow one of the two approaches:
* (Easier) Try to run this code on a machine with a display.
* (Harder) Install `xvfb` (`sudo apt install xvfb`) and run the code with `xvfb-run` with headless mode enabled:
```
xvfb-run -a python demo.py --debug --headless
```

### GPU Out of Memory

Try reducing the `max_depth` and `sampling_rate` in `src/demo.py` file. If the problem persists, try running the code on a machine with a higher VRAM GPU.
```
python demo.py --debug --max_depth 1.5 --sampling_rate 0.5
```