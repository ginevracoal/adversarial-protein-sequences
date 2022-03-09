## Setup

Python version 3.7.6

Install the virtual environment:
```
python -m venv venv
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Download esm1:
```
cd ~/.cache/torch/hub/checkpoints/
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
```