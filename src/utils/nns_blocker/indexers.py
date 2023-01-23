from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import faiss
import numpy as np


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
        from sklearn.neighbors import NearestNeighbors

        self._index = NearestNeighbors(**self.nn_kwargs)
        self._index.fit(data)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        distances, indices = self._index.kneighbors(queries, n_neighbors=k)
        return BatchSearchResult(distances.tolist(), indices.tolist())


class LuceneIndexer(Indexer):
    ...


class FaissIndexer(Indexer):
    def __init__(
        self,
        *,
        index_factory: str = "Flat",
        metric_type: Optional[int] = None,
    ):
        super().__init__()
        self.index_factory = index_factory
        self.metric_type = metric_type

    def build_index(
        self,
        data: np.array,
        *,
        train_size: Optional[int] = None,
        batch_size: int = 1000,
    ):
        size = len(data[0])
        if self.metric_type is None:
            self._index = faiss.index_factory(size, self.index_factory)
        else:
            self._index = faiss.index_factory(
                size, self.index_factory, self.metric_type
            )

        if train_size is not None:
            train_data = data[:train_size]
            self._index.train(train_data)

        for i in range(0, len(data), batch_size):
            batch_data = data[i : i + batch_size]
            self._index.add(batch_data)

    def batch_search(self, queries: np.array, k: int = 10) -> BatchSearchResult:
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self._index.search(queries, k)
        return BatchSearchResult(scores.tolist(), indices.astype(int).tolist())


class NMSLIBIndexer(Indexer):
    ...
