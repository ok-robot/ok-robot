#mamba create -n home_engine python=3.10
#conda create -n home_engine python=3.10
#mamba activate home_engine
#conda activate home_engine
#mamba install -y -c "nvidia/label/cuda-11.8.0" -c pytorch -c nvidia cuda-toolkit pytorch torchvision torchaudio
conda install -y -c "nvidia/label/cuda-11.8.0" -c pytorch -c nvidia cuda-toolkit pytorch torchvision torchaudio
python -m pip install -r requirements.txt
cd clip-fields/gridencoder/
python setup.py install
cd ../..
cd lerf
python -m pip install -e .
ns-install-cli
cd ..
cd usa
pip install -e .
