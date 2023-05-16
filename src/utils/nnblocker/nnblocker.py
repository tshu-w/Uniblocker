from typing import Callable

import pandas as pd
from codetiming import Timer

from src.utils import chunks

from .indexers import Indexer


class NNBlocker:
    def __init__(
        self,
        dfs: list[pd.DataFrame],
        vectorizer: Callable,
        indexer: Indexer,
    ) -> None:
        self.dfs = dfs
        self.vectorizer = vectorizer
        self.indexer = indexer

    def __call__(
        self,
        batch_size: int = 128,
        k: int = 100,
    ) -> list[set[tuple]]:
        with Timer(text="Convert time: {milliseconds:.0f} ms"):
            queries = self.vectorizer(self.dfs[0])
            data = self.vectorizer(self.dfs[-1])
        with Timer(text="Index time: {milliseconds:.0f} ms"):
            self.indexer.build_index(data)

        total_indices = []
        with Timer(text="Total Query time: {milliseconds:.0f} ms"):
            for b_queries in chunks(queries, batch_size):
                _, b_indices = self.indexer.batch_search(b_queries, k=k)
                total_indices.extend(b_indices)

        candidates = []
        flags = set()  # Comparison Propagation
        for i in range(k):
            cands = set()
            for j in range(len(total_indices)):
                if i >= len(total_indices[j]):
                    continue

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
