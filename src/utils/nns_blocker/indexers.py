from abc import ABC, abstractmethod
from collections import Counter
from typing import NamedTuple, Optional

import faiss
import numpy as np
import pyterrier as pt
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


class TerrierIndexer(Indexer):
    def __init__(
        self,
        index_location: str,
        index_kwargs: dict,
        search_kwargs: dict,
    ):
        super().__init__()
        if not pt.started():
            pt.init()
        self.index_location = index_location
        self.index_kwargs = index_kwargs
        self.searcher_kwargs = search_kwargs

    def build_index(self, data):
        self._indexer = pt.IterDictIndexer(self.index_location, **self.index_kwargs)
        data = data.apply(Counter)
        data.reset_index(inplace=True, drop=True)
        data = data.to_frame("toks")
        data.index.names = ["docno"]
        data = data.reset_index()
        data.docno = data.docno.astype("str")
        self._indexer = self._indexer.index(data.to_dict(orient="records"))
        self._searcher = pt.BatchRetrieve(
            self._indexer, num_results=100, **self.searcher_kwargs
        )
        self._searcher = (
            pt.rewrite.tokenise(lambda x: x, matchop=True) >> self._searcher
        )

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        self._searcher[1].controls["end"] = str(k - 1)

        queries.reset_index(inplace=True, drop=True)
        queries = queries.to_frame("query")
        queries.index.names = ["qid"]
        queries = queries.reset_index()
        queries["qid"] = queries["qid"].astype("str")
        results = self._searcher.transform(queries)

        scores, indices = [], []
        lst_qid = -1
        for r in results.itertuples():
            if r.qid != lst_qid:
                scores.append([])
                indices.append([])

            lst_qid = r.qid
            scores[-1].append(r.score)
            indices[-1].append(r.docid)

        return BatchSearchResult(scores, indices)


class FaissIndexer(Indexer):
    def __init__(
        self,
        *,
        indexer_factory: str = "Flat",
        metric_type: Optional[int] = None,
    ):
        super().__init__()
        self.indexer_factory = indexer_factory
        self.metric_type = metric_type

    def build_index(
        self,
        data,
        *,
        train_size: Optional[int] = None,
        batch_size: int = 1000,
    ):
        size = len(data[0])
        if self.metric_type is None:
            self._indexer = faiss.index_factory(size, self.indexer_factory)
        else:
            self._indexer = faiss.index_factory(
                size, self.index_factory, self.metric_type
            )

        if train_size is not None:
            train_data = data[:train_size]
            self._indexer.train(train_data)

        for i in range(0, len(data), batch_size):
            batch_data = data[i : i + batch_size]
            self._indexer.add(batch_data)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self._indexer.search(queries, k)
        return BatchSearchResult(scores.tolist(), indices.astype(int).tolist())


class NMSLIBIndexer(Indexer):
    ...
