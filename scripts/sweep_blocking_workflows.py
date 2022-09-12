import json
import os
import sys
from pathlib import Path
from typing import Literal

import wandb
from jsonargparse import CLI
from ray import air, tune

sys.path.append(str(Path(__file__).parents[1]))

from src.baselines.blocking_workflows import blocking_workflows


def run_blocking_workflows(config):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    data_dir_name = Path(config["data_dir"]).name
    wandb.init(
        project="universal-blocker",
        name=f"blocking_workflows/{data_dir_name}",
        dir=str(Path("results") / "blocking_workflows"),
        config=config,
        tags=["baseline"],
    )
    metrics = blocking_workflows(**config)
    wandb.log(metrics)
    wandb.finish()

    dirpath = Path("results") / "blocking_workflows" / data_dir_name
    dirpath.mkdir(parents=True, exist_ok=True)
    metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2)
    metrics_file = Path(dirpath) / "metrics.json"
    with metrics_file.open("w") as f:
        f.write(metrics_str)


def sweep_blocking_workflows(
    data_dirs: list[str] = [],
    workflows: list[Literal["parameter_free", "default_qgrams"]] = ["default_qgrams"],
):
    data_dirs = data_dirs or [
        str(d)
        for d in (Path(".") / "data" / "blocking").iterdir()
        if d.name not in ["songs", "citeseer-dblp"]
    ]

    param_space = {
        "data_dir": tune.grid_search(data_dirs),
        "workflow": tune.grid_search(workflows),
    }
    tune_config = tune.TuneConfig()
    run_config = air.RunConfig(
        name="blocking_workflows",
        local_dir="results/ray",
        log_to_file=True,
        verbose=1,
    )
    trainable = tune.with_parameters(run_blocking_workflows)
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()


if __name__ == "__main__":
    CLI(sweep_blocking_workflows)
