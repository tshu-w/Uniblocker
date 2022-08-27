import argparse
import json
import multiprocessing
import subprocess
from pathlib import Path
from string import Template

assert Path(".git").exists()
DATA_DIR = Path(".") / "data" / "blocking"
EXP_DIR = Path(".") / "results" / "logs" / "simcse"
EXP_DIR.mkdir(parents=True, exist_ok=True)

EXPT_TMP = Template(
    """{
  "data": {
    "class_path": "src.datamodules.Blocking",
    "init_args": {
      "data_dir": "${data_dir}",
      "index_col": "id",
      "n_neighbors": 100,
      "direction": "forward",
      "batch_size": 256,
      "num_workers": 0,
      "pin_memory": false
    }
  },
  "seed": "${seed}",
  "config": "deepblocker.yaml"
}"""
)
EXPTS = []

for data_dir in DATA_DIR.iterdir():
    if data_dir.name in ["songs", "citeseer-dblp"]:
        continue

    kwargs = {
        "data_dir": data_dir,
        "seed": 1234,
    }
    EXPTS.append(EXPT_TMP.substitute({}, **kwargs))


def argument_parser():
    parser = argparse.ArgumentParser(description="run experiments in parallel")
    parser.add_argument(
        "--fast-dev-run",
        nargs="?",
        type=int,
        default=None,
        const=5,
        help="numbers of fast dev run",
    )
    parser.add_argument(
        "--num-expt", type=int, default=1, help="how many experiments per gpu"
    )
    parser.add_argument(
        "--no-run", action="store_true", help="whether not running command"
    )
    parser.add_argument(
        "--devices", nargs="+", default=["0"], help="availabled devices"
    )

    return parser


def run(exp_args, args):
    worker_id = int(multiprocessing.current_process().name.rsplit("-", 1)[1]) - 1
    device = args.devices[worker_id % len(args.devices)]

    exp_name = str(Path(exp_args["data"]["init_args"]["data_dir"]).name).lower()

    outfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_out.log"
    errfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_err.log"

    if args.fast_dev_run:
        cmd = f"""./run fit \\
        --debug \\
        --config configs/{exp_args['config']} \\
        --seed_everything {exp_args['seed']} \\
        --trainer.devices {device}, --trainer.fast_dev_run {args.fast_dev_run} \\
        --data \"{exp_args['data']}\""""
    else:
        cmd = f"""./run fit \\
        --name deepblocker/{exp_name} \\
        --config configs/{exp_args['config']} \\
        --seed_everything {exp_args['seed']} \\
        --trainer.devices {device}, \\
        --data \"{exp_args['data']}\" \\
        >{outfile} 2>{errfile}
        """

    print(exp_name, cmd, sep="\n")

    if not args.no_run:
        subprocess.call(cmd, shell=True)
        print(f"{exp_name} finished")
        if not args.fast_dev_run:
            (EXP_DIR / exp_name).touch()


if __name__ == "__main__":
    args = argument_parser().parse_args()
    pool = multiprocessing.Pool(processes=len(args.devices) * args.num_expt)

    for expt in EXPTS:
        pool.apply_async(
            run,
            kwds={
                "exp_args": json.loads(expt, strict=False),
                "args": args,
            },
        )

    pool.close()
    pool.join()
