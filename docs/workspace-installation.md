# Workspace Installation
### CUDA 12.*
```
# Environment creation
mamba env create -n ok-robot-env -f ./env-cu121.yml
mamba activate ok-robot-env

# Pointnet setup for anygrasp
cd anygrasp/pointnet2/
python setup.py install
cd ../../

# Additional pip packages isntallation
pip install -r requirements-cu121.txt
pip install --upgrade --no-deps --force-reinstall scikit-learn==1.4.0 
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install graspnetAPI

```

### CUDA 11.*
```
mamba env create -n ok-robot-env -f ./env-cu118.yml
mamba activate ok-robot-env

pip install --upgrade --no-deps --force-reinstall scikit-learn==1.4.0
pip install graspnetAPI

# Setup poincept
cd anygrasp/pointnet2/
python setup.py install
```
