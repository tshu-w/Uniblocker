from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import move_data_to_device
from tqdm import tqdm

from src.utils import chunks

from .vectorizer import Vectorizer


class DenseVectorizer(Vectorizer):
    def __init__(
        self,
        model: torch.nn.Module,
        collate_fn: Callable,
        device: Union[torch.device, str, int] = "cpu",
    ):
        self._model = model
        self._collate_fn = collate_fn
        self._device = device

    @torch.no_grad()
    def __call__(self, data: pd.DataFrame, batch_size: int = 128):
        data = data.fillna("").astype(str)
        records = data.to_dict(orient="records")
        embeddings = []
        for batch in tqdm(chunks(records, batch_size)):
            batch = move_data_to_device(self._collate_fn(batch), self._device)
            b_embeddings = F.normalize(self._model(batch)).to("cpu").numpy()
            embeddings.append(b_embeddings)

        embeddings = np.concatenate(embeddings).astype(np.float32)
        return embeddings
