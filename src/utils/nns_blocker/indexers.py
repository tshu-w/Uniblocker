from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import autofaiss
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
        *,
        index_params: dict,
        device_id: Optional[int] = None,
    ):
        super().__init__()
        self.index_params = index_params
        self.device_id = device_id

    def build_index(
        self,
        data,
        *,
        batch_size: int = 1000,
    ):
        self._indexer, index_infos = autofaiss.build_index(
            data,
            **self.index_params,
        )
        if self.device_id is not None:
            res = faiss.StandardGpuResources()
            self._indexer = faiss.index_cpu_to_gpu(res, self.device_id, self._indexer)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self._indexer.search(queries, k)
        return BatchSearchResult(scores.tolist(), indices.astype(int).tolist())
