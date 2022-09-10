from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from jsonargparse import CLI
from rich import print
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.utils import chunks, evaluate, get_candidates


def convert_df_to_corpus(
    df: pd.DataFrame,
    *,
    sep: str = "\t",
) -> list[str]:
    df = df.fillna("")
    text = df.astype(str).apply(lambda row: sep.join(row), axis=1)
    corpus = text.to_list()

    return corpus


def fit_vectorizer(
    corpus: list[str],
    *,
    preprocessor: Optional[Callable] = None,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (5, 5),
    binary: bool = False,
) -> CountVectorizer:
    vectorizer = CountVectorizer(
        preprocessor=preprocessor,
        analyzer=analyzer,
        ngram_range=ngram_range,
        binary=binary,
    )
    vectorizer.fit(corpus)

    return vectorizer


def build_index(
    corpus: list[str],
    vectorizer: CountVectorizer,
    *,
    n_neighbors: int,
    metric: Union[str, Callable] = "cosine",
) -> NearestNeighbors:
    index = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
    )
    index.fit(vectorizer.transform(corpus))

    return index


def knn_join(
    corpus: list[str],
    vectorizer: CountVectorizer,
    index: NearestNeighbors,
    *,
    n_neighbors: Optional[int] = None,
    chunk_size: int = 128,
) -> list[list[int]]:
    indices_list = []
    for texts in tqdm(chunks(corpus, chunk_size)):
        vectors = vectorizer.transform(texts)
        distances, indices = index.kneighbors(vectors, n_neighbors=n_neighbors)
        assert np.all(np.diff(distances) >= 0)
        indices_list.append(indices)

    indices = np.concatenate(indices_list).tolist()

    return indices


def sparse_join(
    data_dir: str = "./data/blocking/cora",
    index_col: str = "id",
    n_neighbors: int = 100,
    direction: Literal["forward", "reversed", "both"] = "forward",
):
    table_paths = sorted(Path(data_dir).glob("[1-2]*.csv"))

    dfs = [pd.read_csv(p, index_col=index_col) for p in table_paths]
    corpuses = [convert_df_to_corpus(df) for df in dfs]

    matches_path = Path(data_dir) / "matches.csv"
    matches = set(pd.read_csv(matches_path).itertuples(index=False, name=None))

    vectorizers = [fit_vectorizer(corpus) for corpus in corpuses]
    indexes = [
        build_index(corpus, vectorizer, n_neighbors=n_neighbors)
        for corpus, vectorizer in zip(corpuses, vectorizers)
    ]

    if len(corpuses) == 1:
        indices_list = [knn_join(corpuses[0], vectorizers[0], indexes[0])]
    else:
        indices_list = [
            knn_join(corpuses[0], vectorizers[1], indexes[1]),
            knn_join(corpuses[1], vectorizers[0], indexes[0]),
        ]

    candidates = get_candidates(
        dfs,
        indices_list,
        n_neighbors=n_neighbors,
        direction=direction,
    )
    metrics = evaluate(candidates, matches)

    print(metrics)
    return metrics


if __name__ == "__main__":
    CLI(sparse_join)
