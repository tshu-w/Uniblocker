<div align="center">

<h2>Towards Universal Dense Blocking for Entity Resolution</h2>

</div>

## Description
Source code and data for the paper: **Towards Universal Dense Blocking for Entity Resolution**.

<img src="https://user-images.githubusercontent.com/13161779/222436932-23f80a1a-1177-4a08-a8c7-9b1085c50afb.png" width=75%>

<img src="https://user-images.githubusercontent.com/13161779/222437026-13c9dc2c-aecb-413f-8367-bc666c05164f.png" width=75%>

## How to run

First, install the dependencies and download the resources.
```console
# clone project
git lfs install
git clone https://github.com/tshu-w/uniblocker # It will take a while for LFS to download the benchmark data
cd uniblocker
unzip data/blocking.zip -d data

# [SUGGESTED] use conda environment
conda env create -f environment.yaml
conda activate uniblocker

# [ALTERNATIVE] install requirements directly
pip install -r requirements.txt

# [OPTIONAL] download resources
python scripts/download_gittables.py # pre-training corpus
bash scripts/download_fasttext_model.sh # fasttext model for DeepBlocker
```

Next, to obtain the main results of the paper:
```console
# Pre-training
./run --config configs/uniblocker.yaml --config configs/gittables.yaml

# Evaluation
bash scripts/sweep_uniblocker.sh

bash scripts/sweep_deepblocker.sh
bash scripts/sweep_sudowoodo.sh
python scripts/sweep_sparse_join.py
python scripts/sweep_blocking_workflows.py

# Scalability Evaluation
python scripts/scala_prepare.py
for f in scripts/scala_*.py; do python $f ; done
```

You can also run experiments independently using the `run` script.
```console
# fit with the config and cmd line arguments
./run fit --config configs/uniblocker.yaml --config configs/cora.yaml --data.batch_size 32 --trainer.devices 0,

# evaluate with the checkpoint
./run test --config configs/uniblocker.yaml --ckpt_path CKPT_PATH

# get the script help
./run --help
./run fit --help
```

## Benchmark

Details of the constructed benchmark can be found in the README of the data.

## TODO

- [ ] Add Benchmark README
- [ ] Separate [NNBlocker](https://github.com/tshu-w/uniblocker/tree/main/src/utils/nnblocker) into a standalone package to facilitate further research on nearest neighbor blocking techniques.
