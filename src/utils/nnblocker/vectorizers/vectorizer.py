from abc import ABC, abstractmethod

import pandas as pd


class Vectorizer(ABC):
    """wrapper class for various vectorizers."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame): ...
