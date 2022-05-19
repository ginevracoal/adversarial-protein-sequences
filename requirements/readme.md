# Setup 

## Linux CentOS 7 + python 3.8

```
conda create -n conda_venv python=3.8 anaconda
conda create -n conda_venv --file conda_requirements_centos7_python3.8 
conda activate conda_venv
pip install fair-esm==0.4.2
module load conda/4.9.2
module load cuda/11.0.3
```

## Ubuntu 18.10 + python 3.7.6

```
python3.7 -m venv pip_venv
source pip_venv/bin/activate
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r pip_requirements_ubuntu18.10_python3.7.6.txt
```

