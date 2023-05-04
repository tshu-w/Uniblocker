from pathlib import Path
from typing import Optional

import pandas as pd
import torch.nn as nn
from jsonargparse import CLI
from rich import print
from torch.utils.data.dataloader import default_collate

from src.utils import evaluate
from src.utils.nnblocker import DenseVectorizer, FaissIndexer, NNBlocker


def faiss_join(
    model: nn.Module,
    data_dir: str = "./data/blocking/cora",
    size: str = "",
    index_col: str = "id",
    n_neighbors: int = 100,
    device_id: Optional[int] = 0,
    threads: int = 12,
):
    table_paths = sorted(Path(data_dir).glob(f"[1-2]*{size}.csv"))
    dfs = [pd.read_csv(p, index_col=index_col) for p in table_paths]

    collate_fn = getattr(model, "collate_fn", default_collate)
    model = model.to(device_id)

    converter = DenseVectorizer(model, collate_fn, device_id)
    indexer = FaissIndexer(device_id=device_id, threads=threads)
    blocker = NNBlocker(dfs, converter, indexer)
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
