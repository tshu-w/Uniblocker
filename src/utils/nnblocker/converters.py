from typing import Callable

import pandas as pd


class SparseConverter:
    def __init__(
        self,
        tokenizer: Callable,
    ):
        self.tokenizer = tokenizer
        self.is_qgram_tokenizer = (
            hasattr(tokenizer, "__self__")
            and tokenizer.__self__.__class__.__name__ == "QgramTokenizer"
        )

    def __call__(self, data: pd.DataFrame):
        def tokenize_row(row):
            s = " ".join(row).lower()
            if self.is_qgram_tokenizer:
                # replace space with special character like BPE tokenizer
                s = s.replace(" ", "Ġ")
            return self.tokenizer(s)

        data = data.fillna("").astype(str)
        return data.apply(tokenize_row, axis=1)
