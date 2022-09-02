import json
import os
import sys
from pathlib import Path
from typing import Literal

import wandb
from jsonargparse import CLI
from ray import air, tune

sys.path.append(str(Path(__file__).parents[1]))

from src.baselines.sparse_join import sparse_join


def run_sparse_join(config):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    data_dir_name = Path(config["data_dir"]).name
    wandb.init(
        project="universal-blocker",
        name=f"sparse_join/{data_dir_name}",
        dir=str(Path("results") / "sparse_join"),
        config=config,
    )
    metrics = sparse_join(**config)
    wandb.log(metrics)
    wandb.finish()

    dirpath = Path("results") / "sparse_join" / data_dir_name
    dirpath.mkdir(parents=True, exist_ok=True)
    metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2)
    metrics_file = Path(dirpath) / "metrics.json"
    with metrics_file.open("w") as f:
        f.write(metrics_str)


def sweep_sparse_join(
    data_dirs: list[str] = [],
    n_neighbors: list[int] = [100],
    direction: list[Literal["forward", "reversed", "both"]] = ["forward"],
):
    data_dirs = data_dirs or [
        str(d)
        for d in (Path(".") / "data" / "blocking").iterdir()
        if d.name not in ["songs", "citeseer-dblp"]
    ]

    param_space = {
        "data_dir": tune.grid_search(data_dirs),
        "n_neighbors": tune.grid_search(n_neighbors),
        "direction": tune.grid_search(direction),
    }
    tune_config = tune.TuneConfig()
    run_config = air.RunConfig(
        name="sparse_join",
        local_dir="results/ray",
        log_to_file=True,
        verbose=1,
    )
    trainable = tune.with_parameters(run_sparse_join)
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()


if __name__ == "__main__":
    CLI(sweep_sparse_join)