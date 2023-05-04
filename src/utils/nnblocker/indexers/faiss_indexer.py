from math import log2, sqrt
from typing import Optional

import faiss
import numpy as np

from .indexer import BatchSearchResult, Indexer


class FaissIndexer(Indexer):
    def __init__(
        self,
        index_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        device_id: Optional[int] = None,
        threads: Optional[int] = None,
    ):
        super().__init__()
        self.index_factory = index_factory
        self.metric_type = metric_type
        self.device_id = device_id
        if threads is not None:
            faiss.omp_set_num_threads(threads)

    def build_index(
        self,
        data,
        *,
        batch_size: int = 1000,
    ):
        if self.index_factory is None:
            # ref: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
            x_initial = 4 * sqrt(len(data))  # between 4xsqrt(n) and 16xsqrt(n)
            nlist = 2 ** round(log2(x_initial))
            self.index_factory = f"IVF{nlist},Flat"
        else:
            nlist = None

        if self.metric_type is None:
            self._indexer = faiss.index_factory(len(data[0]), self.index_factory)
        else:
            self._indexer = faiss.index_factory(
                len(data[0]), self.index_factory, self.metric_type
            )

        if self.device_id is not None:
            res = faiss.StandardGpuResources()
            self._indexer = faiss.index_cpu_to_gpu(res, self.device_id, self._indexer)

        if nlist is not None:
            self._indexer.nprobe = min(100, nlist)
            self._indexer.train(data)

        for i in range(0, len(data), batch_size):
            batch_data = data[i : i + batch_size]
            self._indexer.add(batch_data)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self._indexer.search(queries, k)
        return BatchSearchResult(scores.tolist(), indices.astype(int).tolist())
