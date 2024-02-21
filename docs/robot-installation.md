# Robot Installation and Setup
**Home Robot Instatallation:** Follow the [home-robot installation instructions](https://github.com/leo20021210/home-robot/blob/main/docs/install_robot.md) to install home-robot on your Stretch robot.

**Copy Hab Stretch Folder:** Copy hab stretch folder from [home robot repo](https://github.com/facebookresearch/home-robot/tree/main/assets/hab_stretch) 
```
cd $OK-Robot/
cp -r home-robot/assets/hab_stretch/ ok-robot-hw
```

## OpenAI installation

Run this command to install the OpenAI APIs if you want to use GPT:
```
pip install openai
```
