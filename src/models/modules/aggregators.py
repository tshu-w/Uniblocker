from abc import ABC, abstractmethod
from typing import Counter, Literal

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import Dataset

AGGREGATOR_TYPE = Literal["average", "sif"]


class Aggregator(ABC):
    def __init__(self, tokenizer, embedder) -> None:
        self.tokenizer = tokenizer
        self.embedder = embedder

    @abstractmethod
    def __call__(self, batch):
        pass


class AverageAggregator(Aggregator):
    def __call__(self, batch: list[dict]):
        embeddings = []
        texts = [" ".join(str(v).casefold() for v in r.values() if v) for r in batch]
        for text in texts:
            embedding_list = [
                self.embedder.get_word_vector(token) for token in self.tokenizer(text)
            ]
            if embedding_list:
                embedding = np.mean(
                    embedding_list,
                    axis=0,
                )
            else:
                embedding = np.empty((self.embedder.get_dimension(),))

            embeddings.append(embedding)

        return torch.Tensor(np.array(embeddings))


class SIFAggregator(Aggregator):
    def __init__(
        self, tokenizer, embedder, sif_weighting_param=1e-3, remove_pc=True, min_freq=0
    ) -> None:
        super().__init__(tokenizer, embedder)

        self.sif_weighting_param = sif_weighting_param
        self.remove_pc = remove_pc
        self.min_freq = min_freq

    def prepare(self, ds: Dataset) -> None:
        token_counter = Counter()
        texts = []
        for idx in range(len(ds)):
            text = " ".join(str(v).casefold() for v in ds[idx].values() if v)
            texts.append(text)
            token_counter.update(self.tokenizer(text))

        token_count = sum(token_counter.values())

        self.token_weight = {}
        a = self.sif_weighting_param
        for token, frequency in token_counter.items():
            if frequency >= self.min_freq:
                self.token_weight[token] = a / (a + frequency / token_count)
            else:
                self.token_weight[token] = 1.0

        embeddings = []
        for text in texts:
            embedding_list = [
                self.token_weight[token] * self.embedder.get_word_vector(token)
                for token in self.tokenizer(text)
            ]
            if embedding_list:
                embedding = np.mean(
                    embedding_list,
                    axis=0,
                )
            else:
                embedding = np.empty((self.embedder.get_dimension(),))

            embeddings.append(embedding)

        if self.remove_pc:
            embeddings = np.array(embeddings)
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(embeddings)
            pc = svd.components_
            assert (embeddings.dot(pc.transpose()) * pc == embeddings @ pc.T * pc).all()
            self.pc = pc

    def __call__(self, batch: list[dict]):
        embeddings = []
        texts = [" ".join(str(v).casefold() for v in r.values() if v) for r in batch]
        for text in texts:
            embedding_list = [
                self.token_weight[token] * self.embedder.get_word_vector(token)
                for token in self.tokenizer(text)
            ]
            if embedding_list:
                embedding = np.mean(
                    embedding_list,
                    axis=0,
                )
            else:
                embedding = np.empty((self.embedder.get_dimension(),))

            embeddings.append(embedding)

        if self.remove_pc:
            embeddings = np.array(embeddings)
            embeddings = embeddings - embeddings @ self.pc.T * self.pc

        return torch.Tensor(np.array(embeddings))


def get_aggregator(type: AGGREGATOR_TYPE) -> Aggregator:
    if type == "average":
        return AverageAggregator
    else:
        return SIFAggregator
