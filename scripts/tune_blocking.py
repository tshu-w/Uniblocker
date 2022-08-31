import itertools
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from jsonargparse import CLI
from pytorch_lightning.utilities import device_parser
from ray import air, tune

sys.path.append(str(Path(__file__).parents[1]))

from src.utils.lit_cli import LitCLI

assert Path(".git").exists()


def run_cli(config, debug: bool = True, command: str = "fit", devices: int = 1):
    # https://github.com/Lightning-AI/lightning/pull/14319
    device_parser.num_cuda_devices = torch.cuda.device_count
    device_parser.is_cuda_available = torch.cuda.is_available
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    exp_name = f"{Path(config['config_file']).stem}/{Path(config['data_dir']).name}"
    data_kwargs = {
        "class_path": "src.datamodules.Blocking",
        "init_args": {
            "data_dir": str(config["data_dir"]),
            "n_neighbors": config["n_neighbors"],
            "direction": config["direction"],
            "batch_size": config["batch_size"],
            "num_workers": 0,
            "pin_memory": True,
        },
    }
    ckpt_path = config["ckpt_path"] or "null"
    data = json.dumps(data_kwargs)

    sys.argv = list(
        itertools.chain(
            ["./run", f"{command}"],
            ["--name", f"{exp_name}"],
            ["--config", f"{config['config_file']}"],
            ["--seed_everything", f"{config['seed']}"],
            ["--trainer.devices", f"{devices}"],
            ["--ckpt_path", f"{ckpt_path}"],
            ["--data", f"{data}"],
            ["--trainer.fast_dev_run", "5", "--debug"] if debug else [],
        )
    )
    print(shlex.join(sys.argv))
    LitCLI(
        parser_kwargs={
            "default_config_files": ["configs/presets/default.yaml"],
        }
    )


def tune_blocking(
    command: Literal["fit", "validate", "test", "tune", "predict"] = "fit",
    debug: bool = False,
    gpus_per_trial: Union[int, float] = 1,
    seed: list[int] = [123],
    ckpt_path: list[Optional[str]] = [None],
    config_file: list[str] = ["configs/simcse.yaml"],
    n_neighbors: list[int] = [100],
    direction: list[Literal["forward", "reversed", "both"]] = ["forward"],
    batch_size: list[int] = [32],
):
    data_dirs = [
        str(d)
        for d in (Path(".") / "data" / "blocking").iterdir()
        if d.name not in ["songs", "citeseer-dblp"]
    ]

    param_space = {
        "seed": tune.grid_search(seed),
        "ckpt_path": tune.grid_search(ckpt_path),
        "config_file": tune.grid_search(config_file),
        "data_dir": tune.grid_search(data_dirs),
        "n_neighbors": tune.grid_search(n_neighbors),
        "direction": tune.grid_search(direction),
        "batch_size": tune.grid_search(batch_size),
    }

    tune_config = tune.TuneConfig(reuse_actors=False)
    run_config = air.RunConfig(
        name="blocking",
        local_dir="results/ray",
        log_to_file=True,
    )
    trainable = tune.with_parameters(
        run_cli,
        debug=debug,
        command=command,
        devices=math.ceil(gpus_per_trial),
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 8, "gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()


def fit(*args, **kwargs):
    tune_blocking(command="fit", *args, **kwargs)


def validate(*args, **kwargs):
    tune_blocking(command="validate", *args, **kwargs)


def test(*args, **kwargs):
    tune_blocking(command="test", *args, **kwargs)


if __name__ == "__main__":
    CLI([fit, validate, test])
