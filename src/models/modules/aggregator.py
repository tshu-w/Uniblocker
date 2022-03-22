from abc import ABC, abstractmethod
from typing import Counter, Literal

import numpy as np
from sklearn.decomposition import TruncatedSVD

AGGREGATOR_TYPE = Literal["average", "sif"]


def serialize(records: list[list[tuple]]) -> list[str]:
    return [" ".join(f"{t[1] or ''}" for t in record) for record in records]


class Aggregator(ABC):
    def __init__(self, tokenizer, embedder) -> None:
        self.tokenizer = tokenizer
        self.embedder = embedder

    @abstractmethod
    def __call__(self, batch):
        pass

    # make Aggregator pickable
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("embedder", None)
        return state


class AverageAggregator(Aggregator):
    def __call__(self, table):
        embeddings = []
        texts = serialize(table["record"])
        for text in texts:
            embeddings.append(
                np.mean(
                    [
                        self.embedder.get_word_vector(token)
                        for token in self.tokenizer(text)
                    ],
                    axis=0,
                )
            )

        return {"embeddings": embeddings}


class SIFAggregator(Aggregator):
    def __init__(
        self, tokenizer, embedder, sif_weighting_param=1e-3, remove_pc=True, min_freq=0
    ) -> None:
        super().__init__(tokenizer, embedder)

        self.sif_weighting_param = sif_weighting_param
        self.remove_pc = remove_pc
        self.min_freq = min_freq

    def __call__(self, table):
        token_counter = Counter()
        texts = serialize(table["record"])
        for text in texts:
            token_counter.update(self.tokenizer(text))
        token_count = sum(token_counter.values())

        token_weight = {}
        a = self.sif_weighting_param
        for token, frequency in token_counter.items():
            if frequency >= self.min_freq:
                token_weight[token] = a / (a + frequency / token_count)
            else:
                token_weight[token] = 1.0

        embeddings = []
        for text in texts:
            embeddings.append(
                np.mean(
                    [
                        token_weight[token] * self.embedder.get_word_vector(token)
                        for token in self.tokenizer(text)
                    ],
                    axis=0,
                )
            )

        if self.remove_pc:
            embeddings = np.array(embeddings)
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(embeddings)
            pc = svd.components_
            assert (embeddings.dot(pc.transpose()) * pc == embeddings @ pc.T * pc).all()
            embeddings = embeddings - embeddings @ pc.T * pc

        return {"embeddings": embeddings}


def get_aggregator(type: AGGREGATOR_TYPE) -> Aggregator:
    if type == "average":
        return AverageAggregator
    else:
        return SIFAggregator
