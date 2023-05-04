from typing import Optional

import nmslib

from .indexer import BatchSearchResult, Indexer


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
