from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import move_data_to_device
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src.utils import chunks


class Converter(ABC):
    """wrapper class for various conveters."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame):
        ...


class CountVectorizerConverter(Converter):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Optional[Callable] = None,
        binary: bool = False,
    ):
        self.vectorizer = CountVectorizer(
            tokenizer=tokenizer,
            binary=binary,
        )

        self.is_qgram_tokenizer = (
            hasattr(tokenizer, "__self__")
            and tokenizer.__self__.__class__.__name__ == "QgramTokenizer"
        )
        df = df.fillna("").astype(str)
        corpus = df.apply(self.preprocess, axis=1).to_list()
        self.vectorizer.fit_transform(corpus)

    def __call__(self, df: pd.DataFrame):
        df = df.fillna("").astype(str)
        corpus = df.apply(self.preprocess, axis=1).to_list()
        return self.vectorizer.transform(corpus)

    def preprocess(self, row):
        s = " ".join(row).lower()
        if self.is_qgram_tokenizer:
            # replace space with special character like BPE tokenizer
            s = s.replace(" ", "Ġ")
        return s


class NeuralConverter(Converter):
    def __init__(
        self, model: nn.Module, collate_fn: Callable, device: Union[str, int] = "cpu"
    ):
        self.model = model
        self.collate_fn = collate_fn
        self.device = device

    @torch.no_grad()
    def __call__(self, df: pd.DataFrame, batch_size: int = 512):
        df = df.fillna("").astype(str)
        records = df.to_dict(orient="records")
        embeddings = []
        for batch in tqdm(chunks(records, batch_size)):
            batch = move_data_to_device(self.collate_fn(batch), self.device)
            b_embeddings = F.normalize(self.model(batch)).to("cpu").numpy()
            embeddings.append(b_embeddings)

        embeddings = np.concatenate(embeddings).astype(np.float32)
        return embeddings
