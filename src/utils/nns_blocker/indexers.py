import itertools
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from runpy import run_module
from typing import NamedTuple, Optional
from unittest.mock import patch

import faiss
import numpy as np
from pyserini.analysis import JWhiteSpaceAnalyzer
from pyserini.search import LuceneSearcher
from sklearn.neighbors import NearestNeighbors

# should left after pyserini import
from jnius import autoclass  # isort: skip

BooleanQuery = autoclass("org.apache.lucene.search.BooleanQuery")
BooleanQuery.setMaxClauseCount(2147483647)


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


class LuceneIndexer(Indexer):
    def __init__(
        self,
        save_dir: str,
        threads: Optional[int] = None,
        index_argv: str = "--keepStopwords --stemmer none --pretokenized",
    ):
        self.save_dir = Path(save_dir)
        self.threads = threads if threads is not None else 1
        self.index_argv = index_argv

    def build_index(self, data):
        data = data.apply(" ".join)
        data.reset_index(inplace=True, drop=True)
        data = data.to_frame("contents")
        data.index.names = ["id"]
        data = data.reset_index()
        data["id"] = data["id"].astype("str")
        corpus_file = self.save_dir / "corpus.jsonl"
        data.to_json(corpus_file, orient="records", lines=True, force_ascii=False)

        index_dir = self.save_dir / "lucene"
        argv = list(
            itertools.chain(
                ["pyserini.index.lucene"],
                ["--collection", "JsonCollection"],
                ["--input", str(self.save_dir)],
                ["--index", str(index_dir)],
                ["--threads", str(self.threads)],
                shlex.split(self.index_argv),
            )
        )
        print(shlex.join(argv))
        with patch("sys.argv", argv):
            run_module(argv[0], run_name="__main__")

        self._searcher = LuceneSearcher(str(index_dir))
        analyzer = JWhiteSpaceAnalyzer()
        self._searcher.set_analyzer(analyzer)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        queries = queries.apply(" ".join).to_list()
        query_ids = list(map(str, range(len(queries))))
        results = self._searcher.batch_search(
            queries, query_ids, k=k, threads=self.threads
        )
        values = [results.get(str(i), []) for i in range(len(queries))]
        scores = [[r.score for r in rl] for rl in values]
        indices = [[int(r.docid) for r in rl] for rl in values]
        return BatchSearchResult(scores, indices)


class FaissIndexer(Indexer):
    def __init__(
        self,
        *,
        index_factory: str = "Flat",
        nprobe: int = 1,
        metric_type: Optional[int] = None,
        device_id: Optional[int] = None,
        threads: Optional[int] = None,
    ):
        super().__init__()
        self.index_factory = index_factory
        self.nprobe = nprobe
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
        size = len(data[0])
        if self.metric_type is None:
            self._indexer = faiss.index_factory(size, self.index_factory)
        else:
            self._indexer = faiss.index_factory(
                size, self.index_factory, self.metric_type
            )

        if self.device_id is not None:
            res = faiss.StandardGpuResources()
            self._indexer = faiss.index_cpu_to_gpu(res, self.device_id, self._indexer)

        self._indexer.nprobe = self.nprobe
        self._indexer.train(data)

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
