from dataclasses import dataclass
from functools import partial
from itertools import product, starmap
from typing import Literal, Optional

import numpy as np

# https://github.com/anhaidgroup/py_stringmatching/issues/80
np.int = int
np.float = float
import py_stringmatching as sm
import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from .augment import Augmenter
from .helpers import serialize


@dataclass
class TransformerCollator:
    tokenizer: PreTrainedTokenizer
    max_length: Optional[int] = None
    serialize_mode: Literal["bare", "full"] = "bare"

    def __call__(
        self,
        batch: list[dict],
    ) -> dict[str, any]:
        _serialize = partial(serialize, mode=self.serialize_mode)
        texts = list(map(_serialize, batch))
        features = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return features


@dataclass
class TransformerCollatorWithAugmenter(TransformerCollator):
    augmenter: Optional[Augmenter] = None

    def __call__(
        self,
        batch: list[dict],
    ) -> dict[str, any]:
        features = super().__call__(batch)
        if self.augmenter is None or not torch.is_grad_enabled():
            return features

        augmented_batch = list(map(self.augmenter, batch))
        _serialize = partial(serialize, mode=self.serialize_mode)
        augmented_texts = list(map(_serialize, augmented_batch))

        augmented_features = self.tokenizer(
            augmented_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return features, augmented_features


@dataclass
class TransformerCollatorWithDistances(TransformerCollatorWithAugmenter):
    def sparse_similarity(
        self,
        s1: str,
        s2: str,
        similarity_measures=[
            sm.similarity_measure.cosine.Cosine(),
            sm.similarity_measure.dice.Dice(),
            sm.similarity_measure.jaccard.Jaccard(),
            sm.similarity_measure.overlap_coefficient.OverlapCoefficient(),
            sm.similarity_measure.tversky_index.TverskyIndex(),
        ],
    ) -> float:
        t1 = set(self.tokenizer.tokenize(s1))
        t2 = set(self.tokenizer.tokenize(s2))
        score = 0
        for similarity_measure in similarity_measures:
            sim_score = similarity_measure.get_sim_score(t1, t2)
            if sim_score > score:
                score = sim_score
        return score

    def __call__(
        self,
        batch: list[dict],
    ) -> dict[str, any]:
        if not torch.is_grad_enabled():
            return super().__call__(batch)

        features = super().__call__(batch)
        _serialize = partial(serialize, mode=self.serialize_mode)
        texts = list(map(_serialize, batch))
        batch_size = len(texts)
        distances = list(starmap(self.sparse_similarity, product(texts, texts)))
        distances = torch.Tensor(
            [
                distances[i * batch_size : i * batch_size + batch_size]
                for i in range(batch_size)
            ]
        )
        return features, distances


@dataclass
class RetroMAECollator(TransformerCollator, DataCollatorForLanguageModeling):
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            )

    def __call__(
        self,
        batch: list[dict],
    ) -> dict[str, any]:
        if not torch.is_grad_enabled():
            return super().__call__(batch)

        _serialize = partial(serialize, mode=self.serialize_mode)
        texts = list(map(_serialize, batch))
        feature = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoder_input_ids = feature["input_ids"].clone()
        self.mlm_probability = self.encoder_mlm_probability
        encoder_input_ids, encoder_labels = self.torch_mask_tokens(encoder_input_ids)
        encoder_attention_mask = feature["attention_mask"].clone()

        decoder_input_ids = feature["input_ids"].clone()
        decoder_labels = decoder_input_ids.clone()
        # only compute loss on not special tokens
        for special_token_id in self.tokenizer.all_special_ids:
            decoder_labels[decoder_labels == special_token_id] = -100

        attention_mask = encoder_attention_mask.clone()
        probability_matrix = (
            attention_mask.unsqueeze(2) @ attention_mask.unsqueeze(1)
        ).float()
        probability_matrix[:, :, 1:] *= 1 - self.decoder_mlm_probability
        diagonal_mask = torch.eye(self.max_length).repeat(attention_mask.size(0), 1, 1)
        probability_matrix[diagonal_mask.bool()] = 0.0
        decoder_attention_mask = torch.bernoulli(probability_matrix)

        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_labels": encoder_labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_labels": decoder_labels,
        }
