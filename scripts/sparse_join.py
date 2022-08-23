import subprocess
from pathlib import Path


def main():
    exexcluded = ["songs", "citeseer-dblp"]
    for data_dir in Path("./data/blocking").iterdir():
        if data_dir.name in exexcluded:
            continue
        dirpath = Path("results") / "sparse_join" / data_dir.name
        dirpath.mkdir(parents=True, exist_ok=True)
        cmd = f"python -m src.baselines.sparse_join save_metrics --data_dir {data_dir} {dirpath}"
        print(cmd)
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
