from math import sqrt
from pathlib import Path

import pandas as pd
import torch.nn as nn
from jsonargparse import CLI
from rich import print
from torch.utils.data.dataloader import default_collate

from src.utils import evaluate
from src.utils.nns_blocker import FaissIndexer, NeuralConverter, NNSBlocker


def faiss_join(
    model: nn.Module,
    data_dir: str = "./data/blocking/cora",
    size: str = "",
    index_col: str = "id",
    n_neighbors: int = 100,
    device_id: int = 0,
    threads: int = 12,
):
    table_paths = sorted(Path(data_dir).glob(f"[1-2]*{size}.csv"))
    dfs = [pd.read_csv(p, index_col=index_col) for p in table_paths]

    collate_fn = getattr(model, "collate_fn", default_collate)
    model = model.to(device_id)

    # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    nlist = int(4 * sqrt(len(dfs[-1])))
    index_factory = f"IVF{nlist},Flat"
    nprobe = min(100, nlist)

    converter = NeuralConverter(model, collate_fn, device_id)
    indexer = FaissIndexer(
        index_factory=index_factory, nprobe=nprobe, device_id=device_id, threads=threads
    )
    blocker = NNSBlocker(dfs, converter, indexer)
    candidates = blocker(k=n_neighbors)

    if size != "":
        # shortcut for scalability experiments
        return

    matches_path = Path(data_dir) / "matches.csv"
    matches = set(pd.read_csv(matches_path).itertuples(index=False, name=None))
    metrics = evaluate(candidates, matches)

    print(metrics)
    return metrics


if __name__ == "__main__":
    CLI(faiss_join)
