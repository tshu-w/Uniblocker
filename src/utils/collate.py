from typing import Optional

from transformers import PreTrainedTokenizer


class TransformerCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        batch: list,
    ):
        texts = [" ".join(t[1] for t in r) for r in batch]
        features = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return features
