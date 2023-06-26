python scripts/sweep_blocking.py test --config_files "[configs/uniblocker.yaml]" --ckpt_paths "[results/fit/uniblocker/gittables/wuxm10ou/checkpoints/step=800-AP=0.53218.ckpt]" --gpus_per_trial 1

# Ablation Studies
# python scripts/sweep_blocking.py test --config_files "[configs/uniblocker.yaml]" --ckpt_paths "[results/fit/uniblocker/blockings/fypnbctx/checkpoints/step=7300-AP=0.47152.ckpt]" --gpus_per_trial 1 # w/o ID
# python scripts/sweep_blocking.py test --config_files "[configs/simcse.yaml]" --ckpt_paths "[results/fit/simcse/gittables/s37eegy6/checkpoints/step=800-AP=0.50508.ckpt]" --gpus_per_trial 1 # w/o DD
# python scripts/sweep_blocking.py test --config_files "[configs/uniblocker.yaml]" --ckpt_paths "[/home/tianshu2020/uniblocker/results/fit/uniblocker/gittables/4eksfe7u/checkpoints/step=2100-AP=0.50440.ckpt]" # w/o FP
# python scripts/sweep_blocking.py test --config_files "[configs/uniblocker.yaml]" --ckpt_paths "[results/fit/uniblocker/gittables/j03mw6oc/checkpoints/step=6300-AP=0.52738.ckpt]" # w/o LP

# archived
# python scripts/sweep_blocking.py test --config_files "[configs/uniblocker.yaml]" --ckpt_paths "[results/fit/uniblocker/blockings/wn102zjx/checkpoints/step=2100-AP=0.53568.ckpt]" --gpus_per_trial 1
# python scripts/sweep_blocking.py fit --config_files "[configs/uniblocker.yaml]" --batch_sizes "[64]" --ckpt_paths "[results/fit/uniblocker/gittables/7d2m6asi/checkpoints/step=8100-AP=0.49128.ckpt]" --gpus_per_trial 1
