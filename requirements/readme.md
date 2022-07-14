# Setup 

## Linux CentOS 7 + python 3.8

```
module load conda/4.9.2
conda create -n esm python=3.8 anaconda
conda create -n esm --file conda_requirements_centos7_python3.8 
conda activate esm
pip install fair-esm==0.4.2
```

## Ubuntu 18.10 + python 3.7.6

```
python3.7 -m venv esm
source esm/bin/activate
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r pip_requirements_ubuntu18.10_python3.7.6.txt
```

## conda environment for structure prediction

Load this enviroment to execute `predict_structures.py`

```
wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh
chmod +x install_colabbatch_linux.sh
./install_colabbatch_linux.sh
conda activate ~/colabfold_batch/colabfold-conda/
conda install -c conda-forge google-colab
conda install -c conda-forge openmm=7.5.1 pdbfixer
conda install -c anaconda seaborn
```