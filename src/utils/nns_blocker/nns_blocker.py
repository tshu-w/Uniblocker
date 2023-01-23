from typing import Callable

import numpy as np
import pandas as pd

from src.utils import chunks

from .indexers import Indexer


class NNSBlocker:
    def __init__(
        self,
        dfs: list[pd.DataFrame],
        converter: Callable,
        indexer: Indexer,
    ) -> None:
        self.dfs = dfs
        self.converter = converter
        self.indexer = indexer

    def __call__(
        self,
        batch_size: int = 128,
        k: int = 100,
    ) -> list[set[tuple]]:
        queries = self.converter(self.dfs[0])
        data = self.converter(self.dfs[-1])
        self.indexer.build_index(data)

        total_indices = []
        for b_queries in chunks(queries, batch_size):
            b_scores, b_indices = self.indexer.batch_search(b_queries, k=k)
            assert np.all(np.diff(b_scores) >= 0)
            total_indices.extend(b_indices)

        candidates = []
        flags = set()  # Comparison Propagation
        for i in range(len(total_indices[0])):
            cands = set()
            for j in range(len(total_indices)):
                ind1 = j
                ind2 = total_indices[j][i]
                if len(self.dfs) == 1 and ind1 > ind2:
                    ind1, ind2 = ind2, ind1

                id1 = self.dfs[0].index[ind1]
                id2 = self.dfs[-1].index[ind2]
                pair = id1, id2
                if id1 != id2 and pair not in flags:
                    cands.add(pair)
                    flags.add(pair)

            candidates.append(cands)

        return candidates
