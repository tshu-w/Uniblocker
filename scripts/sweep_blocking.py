import itertools
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Literal, Optional, Union
from unittest.mock import patch

import ray
from jsonargparse import CLI
from ray import air, tune

sys.path.append(str(Path(__file__).parents[1]))

from src.utils.lit_cli import lit_cli

os.environ["PL_DISABLE_FORK"] = "1"
ray.init(
    _temp_dir=str(Path.home() / ".cache" / "ray"), num_cpus=min(os.cpu_count(), 32)
)


def run_cli(config, debug: bool = True, command: str = "fit", devices: int = 1):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    data_kwargs = {
        "class_path": "src.datamodules.Blocking",
        "init_args": {
            "data_dir": config["data_dir"],
            "batch_size": config["batch_size"],
            "num_workers": 16,
            "pin_memory": True,
        },
    }
    ckpt_path = config["ckpt_path"] or "null"
    data = json.dumps(data_kwargs)

    argv = list(
        itertools.chain(
            ["./run", f"{command}"],
            ["--config", f"{config['config_file']}"],
            ["--seed_everything", f"{config['seed']}"],
            ["--trainer.devices", f"{devices}"],
            ["--ckpt_path", f"{ckpt_path}"],
            ["--data", f"{data}"],
            ["--trainer.fast_dev_run", "5"] if debug else [],
        )
    )
    print(shlex.join(argv))
    with patch("sys.argv", argv):
        lit_cli()

    import wandb

    wandb.finish()


def sweep_blocking(
    command: Literal["fit", "validate", "test", "tune", "predict"] = "fit",
    debug: bool = False,
    gpus_per_trial: Union[int, float] = 1,
    seed: list[int] = [123],
    ckpt_paths: list[Optional[str]] = [None],
    config_files: list[str] = ["configs/uniblocker.yaml"],
    data_dirs: list[str] = [],
    batch_sizes: list[int] = [32],
):
    data_dirs = data_dirs or [
        str(d)
        for d in (Path(".") / "data" / "blocking").iterdir()
        if d.name not in ["songs", "citeseer-dblp"]
    ]

    param_space = {
        "seed": tune.grid_search(seed),
        "ckpt_path": tune.grid_search(ckpt_paths),
        "config_file": tune.grid_search(config_files),
        "data_dir": tune.grid_search(data_dirs),
        "batch_size": tune.grid_search(batch_sizes),
    }

    tune_config = tune.TuneConfig()
    run_config = air.RunConfig(
        name="blocking",
        storage_path=Path("./results/ray").resolve(),
        verbose=1,
    )
    trainable = tune.with_parameters(
        run_cli,
        debug=debug,
        command=command,
        devices=math.ceil(gpus_per_trial),
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()


def fit(*args, **kwargs):
    sweep_blocking(command="fit", *args, **kwargs)


def validate(*args, **kwargs):
    sweep_blocking(command="validate", *args, **kwargs)


def test(*args, **kwargs):
    sweep_blocking(command="test", *args, **kwargs)


if __name__ == "__main__":
    CLI([fit, validate, test])
