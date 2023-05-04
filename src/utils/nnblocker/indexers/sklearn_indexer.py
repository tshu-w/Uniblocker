from sklearn.neighbors import NearestNeighbors

from .indexer import BatchSearchResult, Indexer


class SklearnIndexer(Indexer):
    def __init__(
        self,
        init_kwargs: dict,
    ):
        super().__init__()
        self.init_kwargs = init_kwargs

    def build_index(self, data):
        self._indexer = NearestNeighbors(**self.init_kwargs)
        self._indexer.fit(data)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        n_neighbors = min(k, self._indexer.n_samples_fit_)
        distances, indices = self._indexer.kneighbors(queries, n_neighbors=n_neighbors)
        return BatchSearchResult(distances.tolist(), indices.tolist())
