python scripts/sweep_blocking.py test --config_files "[configs/recordformer.yaml]" --ckpt_paths "[results/fit/recordformer/gittables/7d2m6asi/checkpoints/step=8100-AP=0.49128.ckpt]" --gpus_per_trial 1
# python scripts/sweep_blocking.py fit --config_files "[configs/recordformer.yaml]" --batch_sizes "[64]" --ckpt_paths "[results/fit/recordformer/gittables/7d2m6asi/checkpoints/step=8100-AP=0.49128.ckpt]" --gpus_per_trial 1