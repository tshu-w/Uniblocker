import itertools
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Literal, Optional, Union

from jsonargparse import CLI
from ray import air, tune

sys.path.append(str(Path(__file__).parents[1]))

from src.utils.lit_cli import LitCLI

assert Path(".git").exists()
os.environ["PL_DISABLE_FORK"] = "1"


def run_cli(config, debug: bool = True, command: str = "fit", devices: int = 1):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    exp_name = f"{Path(config['config_file']).stem}/{Path(config['data_dir']).name}"
    data_kwargs = {
        "class_path": "src.datamodules.Blocking",
        "init_args": {
            "data_dir": config["data_dir"],
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
            ["--debug"] if debug else [],
        )
    )
    print(shlex.join(sys.argv))
    LitCLI(
        parser_kwargs={
            cmd: {
                "default_config_files": ["configs/presets/default.yaml"],
            }
            for cmd in ["fit", "validate", "test"]
        },
        save_config_overwrite=True,
    )

    import wandb

    wandb.finish()


def sweep_blocking(
    command: Literal["fit", "validate", "test", "tune", "predict"] = "fit",
    debug: bool = False,
    gpus_per_trial: Union[int, float] = 1,
    seed: list[int] = [123],
    ckpt_paths: list[Optional[str]] = [None],
    config_files: list[str] = ["configs/simcse.yaml"],
    data_dirs: list[str] = [],
    n_neighbors: list[int] = [100],
    directions: list[Literal["forward", "reversed", "both"]] = ["forward"],
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
        "n_neighbors": tune.grid_search(n_neighbors),
        "direction": tune.grid_search(directions),
        "batch_size": tune.grid_search(batch_sizes),
    }

    tune_config = tune.TuneConfig()
    run_config = air.RunConfig(
        name="blocking",
        local_dir="results/ray",
        log_to_file=True,
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
