from abc import ABC, abstractmethod
from typing import NamedTuple


class SearchResult(NamedTuple):
    scores: list[float]
    indices: list[int]


class BatchSearchResult(NamedTuple):
    batch_scores: list[list[float]]
    batch_indices: list[list[int]]


class Indexer(ABC):
    """Wrapper class for various indexers."""

    @abstractmethod
    def build_index(self, data): ...

    def search(self, query, k: int = 10) -> SearchResult:
        batch_scores, batch_indices = self.batch_search([query], k)
        return SearchResult(batch_scores[0], batch_indices[0])

    @abstractmethod
    def batch_search(self, queries, k: int = 10) -> BatchSearchResult: ...
