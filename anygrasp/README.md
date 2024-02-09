<!-- <img src="https://user-images.githubusercontent.com/12446953/208367719-4ef7922f-4001-41f7-aa9f-076e462d1325.png" width="60%"> -->

# Manipulation Module
This section code contains manipulation part of the project. On a high level it takes the RGBD images and a query for the object then using [anygrasp](https://arxiv.org/abs/2212.08333) it generates all possible poses in the 

![Manipulation Pipeline](https://drive.google.com/file/d/1H7ddROUyjqFDhEMOOyr_-vcOqFwm203o/view?usp=sharing)

## Installation
1. Create Conda environment and install torch
```bash
    conda create --name any_grasp python=3.8
    conda activate any_grasp
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

1. Install Minkowski Engine
```bash
    pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine
```

2. Install other requirements from Pip.
```bash
    # export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True # Run this if you facing issues with sklearn install inside graspnertAPI
    pip install -r requirements.txt
```

3. Install ``pointnet2`` module.
```bash
    cd pointnet2
    python setup.py install
```

4. Install [langSAM](https://github.com/luca-medeiros/lang-segment-anything).
```bash
    pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```

5. Installation issues
```bash
    AttributeError: module 'numpy' has no attribute 'float'. -> Make sure the installed numpy version 1.23.0
```

## License Registration
   
Due to the IP issue, currently only the SDK library file of AnyGrasp is available in a licensed manner. Please get the feature id of your machine and fill in the [form](https://forms.gle/XVV3Eip8njTYJEBo6) to apply for the license. After obtaining license follow below instructions

1. Copy `gsnet.*.so` and `lib_cxx.*.so` to this folder according to your Python version (Python>=3.6,<=3.9 is supported). For example, if you use Python 3.6, you can do as follows:
```bash
    cp gsnet_versions/gsnet.cpython-36m-x86_64-linux-gnu.so gsnet.so
    cp ../license_registration/lib_cxx_versions/lib_cxx.cpython-36m-x86_64-linux-gnu.so lib_cxx.so
```

2. Unzip your license and put the folder here as `license`. Refer to [license_registration/README.md](./license_registration/README.md) if you have not applied for license.

3. Put model weights under ``checkpoints/``.

## Demo Code
Run your code like `demo.py` or any desired applications that uses `gsnet.so`. 
```bash
    cd src/; 
    python demo.py --debug
    # For just testing grasping remove the open_communication option in demo.sh. 
```
