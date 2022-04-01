import argparse
import json
import multiprocessing
import subprocess
from pathlib import Path
from string import Template

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "deepmatcher"
EXP_DIR = PROJECT_DIR / "results" / "logs" / "clencoder"
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ARGS = {
    "dataset": "Structured/Walmart-Amazon",
}
EXPT_TMP = Template(
    """{
  "data": {
    "class_path": "src.datamodules.matching.Matching",
    "init_args": {
      "data_dir": "./data/deepmatcher/",
      "dataset": "${dataset}",
      "batch_size": 64,
      "num_workers": 0
    }
  },
  "seed": "${seed}",
  "config": "clencoder.yaml",
  "ckpt_path": "${ckpt_path}"
}"""
)
EXPTS = []

for dir in ["Structured", "Dirty", "Textual"]:
    for dataset in (DATA_DIR / dir).iterdir():
        ckpt_path = "results/fit/cl_pretrain_gittables/03-31T075246/checkpoints/epoch=0-step=1400.ckpt"
        ckpt_path = next(
            (
                Path("./results")
                / "fit"
                / "clencoder"
                / f"{str(dataset.relative_to(DATA_DIR)).lower().replace('/', '_')}"
            ).rglob("*.ckpt")
        )
        print(ckpt_path)
        kwargs = {
            "dataset": str(dataset.relative_to(DATA_DIR)),
            "seed": 123,
            "ckpt_path": ckpt_path,
        }
        EXPTS.append(EXPT_TMP.substitute(DEFAULT_ARGS, **kwargs))


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
    parser.add_argument("--gpus", nargs="+", default=["0"], help="availabled gpus")

    return parser


def run(exp_args, args):
    worker_id = int(multiprocessing.current_process().name.rsplit("-", 1)[1]) - 1
    gpu = args.gpus[worker_id % len(args.gpus)]

    exp_name = str(exp_args["data"]["init_args"]["dataset"]).lower().replace("/", "_")

    outfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_out.log"
    errfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_err.log"

    if args.fast_dev_run:
        cmd = f"""./run test \\
        --debug \\
        --config configs/{exp_args['config']} \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, --trainer.fast_dev_run {args.fast_dev_run} \\
        --ckpt_path '{exp_args['ckpt_path']}' \\
        --data '{exp_args['data']}'"""
    else:
        cmd = f"""./run test \\
        --name clencoder/{exp_name} \\
        --config configs/{exp_args['config']} \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, \\
        --data '{exp_args['data']}' \\
        --ckpt_path '{exp_args['ckpt_path']}' \\
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
    pool = multiprocessing.Pool(processes=len(args.gpus) * args.num_expt)

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
