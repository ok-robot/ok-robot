# Robot Installation and Setup
**Home Robot Installation:** Our hardware controller code relies on [home-robot](https://github.com/facebookresearch/home-robot). Follow these [instructions](https://github.com/facebookresearch/home-robot/blob/main/docs/install_robot.md) to install home-robot on your Stretch Robot.

To check whether home-robot is installed properly and to get familiar with running home-robot based codes, we recommend you try to run [these test scripts](https://github.com/facebookresearch/home-robot/blob/main/tests/hw_manual_test.py).

**Copy Hab Stretch Folder:** Copy hab stretch folder from [home robot repo](https://github.com/facebookresearch/home-robot/tree/main/assets/hab_stretch) 
```
cd $OK-Robot/
cp -r home-robot/assets/hab_stretch/ ok-robot-hw
```

## OpenAI Installation

If you want to use the GPT-client to interact with OK-Robot in a conversational manner, you will need to install the OpenAI APIs.
Run this command to install the OpenAI API client:
```
pip install openai
```

Then, on the OpenAI website, create an API key and set it as an environment variable:
```
export OPENAI_API_KEY="your-api-key"
```

## Urdf Setup
We mainly use `ok-robot-hw/hab_stretch/urdf/stretch_manip_mode.urdf` for operating the robot. The default urdf might not be accurately calibrated for your robot. We suggest either use existing urdf or generate a new urdf following these [instructions](../docs/robot-calibration.md). 

If you are using a exisiting urdf, Make sure
* It contains `wrist_pitch` and `wrist_roll` joints. If not, install dexwrist for [re1](https://docs.hello-robot.com/0.2/stretch-hardware-guides/docs/dex_wrist_guide_re1/) or [re2](https://docs.hello-robot.com/0.2/stretch-hardware-guides/docs/dex_wrist_guide_re2/)


* You add the following joints to default stretch urdf
```
<link name="fake_link_x">
    <inertial>
      <origin rpy="0.0 0.0 0." xyz="0. 0. 0."/>
      <mass value="0.749143203376"/>
      <inertia ixx="0.0709854511955" ixy="-0.00433428742758" ixz="-0.000186110788698" iyy="0.000437922053343" iyz="-0.00288788257713" izz="0.0711048085017"/>
    </inertial>
  </link>
  <joint name="joint_fake" type="prismatic">
    <origin rpy="0. 0. 0." xyz="0. 0. 0."/>
    <axis xyz="1.0 0.0 0.0"/>
    <parent link="base_link"/>
    <child link="fake_link_x"/>
    <limit effort="100.0" lower="-1.0" upper="1.1" velocity="1.0"/>
  </joint>
```
  
* Also, modify the `parent link` of `joint_mast` joint from `base_link` to `fake_link_x`. The joint should finally look like this
```
<joint name="joint_mast" type="fixed">
    <origin xyz="-0.06886239813360509 0.13725755297906447 0.025143215009302215" rpy="1.5725304449603004 0.0027932103811125764 0.013336895699597295"/>
    <axis xyz="0.0 0.0 0.0"/>
    <parent link="fake_link_x"/>
    <child link="link_mast"/>
  </joint>
```

* Then replace the `ok-robot-hw/hab_stretch/urdf/stretch_manip_mode.urdf` with your latest modified urdf.