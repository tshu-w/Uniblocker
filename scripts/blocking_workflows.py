import json
import sys
from pathlib import Path

import wandb

sys.path.append(str(Path(__file__).parents[1]))

from src.baselines.blocking_workflows import blocking_workflows


def main():
    excluded_dirs = ["songs", "citeseer-dblp"]
    for workflow in ["parameter_free", "default_qgrams"]:
        for data_dir in Path("./data/blocking").iterdir():
            if data_dir.name in excluded_dirs:
                continue

            try:
                init_kwargs = {
                    "data_dir": str(data_dir),
                    "workflow": workflow,
                }
                wandb.init(
                    project="universal-blocker",
                    name=f"blocking_workflows/{data_dir.name}",
                    dir=str(Path("results") / "blocking_workflows"),
                    config=init_kwargs,
                )
                metrics = blocking_workflows(**init_kwargs)
                wandb.log(metrics)
                wandb.finish()

                dirpath = (
                    Path("results") / "blocking_workflows" / workflow / data_dir.name
                )
                dirpath.mkdir(parents=True, exist_ok=True)
                metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2)
                metrics_file = Path(dirpath) / "metrics.json"
                with metrics_file.open("w") as f:
                    f.write(metrics_str)
            except Exception:
                pass


if __name__ == "__main__":
    main()
