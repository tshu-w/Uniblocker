import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .vectorizer import Vectorizer


class SparseVectorizer(Vectorizer):
    def __init__(
        self,
        data: pd.DataFrame,
        vectorizer_class=CountVectorizer,
        vectorizer_kwargs: dict = {},
    ):
        self._vectorizer = vectorizer_class(**vectorizer_kwargs)
        self._vectorizer.fit_transform(self._preprocess(data))

    def __call__(self, data: pd.DataFrame):
        return self._vectorizer.transform(self._preprocess(data))

    def _preprocess(self, data: pd.DataFrame):
        data = data.fillna("").astype(str)
        data = data.apply(lambda row: " ".join(row), axis=1).to_list()
        return data
