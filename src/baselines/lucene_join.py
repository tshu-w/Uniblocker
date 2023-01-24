import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from jsonargparse import CLI
from rich import print

# https://github.com/anhaidgroup/py_stringmatching/issues/80
np.int = int
np.float = float
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer

from src.utils import evaluate
from src.utils.nns_blocker import LuceneIndexer, NNSBlocker, SparseConverter


def lucene_join(
    data_dir: str = "./data/blocking/cora",
    size: str = "",
    index_col: str = "id",
    tokenizer: Optional[Callable] = None,
    n_neighbors: int = 100,
):
    table_paths = sorted(Path(data_dir).glob(f"[1-2]*{size}.csv"))
    dfs = [pd.read_csv(p, index_col=index_col) for p in table_paths]

    if tokenizer is None:
        tokenizer = WhitespaceTokenizer().tokenize

    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SparseConverter(tokenizer)
        indexer = LuceneIndexer(save_dir=tmpdir)
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
    CLI(lucene_join)
