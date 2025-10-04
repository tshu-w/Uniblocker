python scripts/sweep_blocking.py test --config_files "[configs/qwen3-embedding-0.6B.yaml]" --batch_sizes "[16]" --seed "[1234]" --gpus_per_trial 1
python scripts/sweep_blocking.py test --config_files "[configs/qwen3-embedding-4B.yaml]" --batch_sizes "[16]" --seed "[1234]" --gpus_per_trial 1
python scripts/sweep_blocking.py test --config_files "[configs/qwen3-embedding-8B.yaml]" --batch_sizes "[16]" --seed "[1234]" --gpus_per_trial 1
