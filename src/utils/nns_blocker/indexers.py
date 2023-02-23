from abc import ABC, abstractmethod
from math import log2, sqrt
from typing import NamedTuple, Optional

import faiss
import nmslib
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SearchResult(NamedTuple):
    scores: list[float]
    indices: list[int]


class BatchSearchResult(NamedTuple):
    batch_scores: list[list[float]]
    batch_indices: list[list[int]]


class Indexer(ABC):
    """Wrapper class for various indexers."""

    @abstractmethod
    def build_index(self, data):
        ...

    def search(self, query, k: int = 10) -> SearchResult:
        batch_scores, batch_indices = self.batch_search([query], k)
        return SearchResult(batch_scores[0], batch_indices[0])

    @abstractmethod
    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        ...


class SklearnIndexer(Indexer):
    def __init__(
        self,
        **nn_kwargs,
    ):
        super().__init__()
        self.nn_kwargs = nn_kwargs

    def build_index(self, data):
        self._indexer = NearestNeighbors(**self.nn_kwargs)
        self._indexer.fit(data)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        n_neighbors = min(k, self._indexer.n_samples_fit_)
        distances, indices = self._indexer.kneighbors(queries, n_neighbors=n_neighbors)
        return BatchSearchResult(distances.tolist(), indices.tolist())


class NMSLIBIndexer(Indexer):
    def __init__(
        self,
        init_kwargs: dict,
        index_params: dict,
        query_params: dict,
        threads: Optional[int] = None,
    ):
        super().__init__()
        self.init_kwargs = init_kwargs
        self.index_params = index_params
        self.query_params = query_params
        self.threads = threads

    def build_index(self, data):
        self._indexer = nmslib.init(**self.init_kwargs)
        self._indexer.addDataPointBatch(data)
        self._indexer.createIndex(self.index_params)
        self._indexer.setQueryTimeParams(self.query_params)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        results = self._indexer.knnQueryBatch(queries, k=k, num_threads=self.threads)
        indices = [r[0].tolist() for r in results]
        scores = [r[1].tolist() for r in results]
        return BatchSearchResult(scores, indices)


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
            # see: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
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
