#!/usr/bin/env bash
mamba create -n home_engine python=3.10
mamba activate home_engine
#mamba install -y -c "nvidia/label/cuda-11.8.0" -c pytorch -c nvidia cuda-toolkit pytorch torchvision torchaudio
mamba install -y -c nvidia -c pytorch cuda-toolkit pytorch torchvision torchaudio
python -m pip install -r requirements.txt
