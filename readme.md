## Basic usage

To install the virtual environment follow the instructions from `requirements/readme.md`

Load modules and activate the venv:
```
module load conda/4.9.2
conda activate esm
``` 

### Set args

- `--data_dir=DATA_DIR` input data directory
- `--out_dir=OUT_DIR` output data directory
- `--dataset=DATASET` dataset filename
- optionally cut sequences to max number of tokens `--max_tokens` (default is `None` )
- optionally choose a maximum number of sequences `--n_sequences` (`None` keeps all the available sequences)
- `--min_filter` sets the minimum number of sequences used to build the reference filtered MSA for each input sequence
- `--n_substitutions` is the number of desired token substitutions in the perturbed sequences
- `--device` sets the desired running device (choose `cuda` or `cpu`)

### Transformers attack examples

Attack single sequence model:
```
cd src/
python attack_single_sequences.py --data_dir=DATA_DIR --out_dir=OUT_DIR --dataset=fastaPF00001 --max_tokens=200 \
	--n_sequences=100 --n_substitutions=3 --device=cuda
```

Attack MSA model:
```
cd src/
./hhfilter.sh
python attack_msa.py --data_dir=DATA_DIR --out_dir=OUT_DIR --dataset=PF00533  \
	--n_sequences=100 --min_filter=100 --n_substitutions=3 --device=cuda
```