# Generating new calibarted URDF
If you already have a calibrated urdf you can skip the following steps for getting new calibrated URDF

* Follow [home-robot calibration instructions](https://github.com/hello-robot/stretch_ros/blob/noetic/stretch_description/README.md#changing-the-tool) to create a new calibrated urdf for your robot.

* Ensure the generated urdf has `wrist_roll` and `wrist_pitch` joints. If not, follow these documentations for [re1](https://docs.hello-robot.com/0.2/stretch-hardware-guides/docs/dex_wrist_guide_re1/) and [re2](https://docs.hello-robot.com/0.2/stretch-hardware-guides/docs/dex_wrist_guide_re2/) hello stretch for installating a dex wrist on the robot. 

* Once you have a proper urdf add the following link and joint to the urdf
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

**URDF Location:** After that replace the `strech_manip_mode.urdf` in `ok-robot-hw/hab_stretch/urdf/` directory with this new calibrated urdf.