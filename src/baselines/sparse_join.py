from pathlib import Path
from typing import Callable, Optional, Union

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
    text = df.astype(str).apply(lambda row: sep.join(row).lower(), axis=1)
    corpus = text.to_list()

    return corpus


def fit_vectorizer(
    corpus: list[str],
    *,
    preprocessor: Optional[Callable] = None,
    tokenizer: Optional[Callable] = None,
    binary: bool = False,
) -> CountVectorizer:
    vectorizer = CountVectorizer(
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        analyzer="word" if tokenizer else "char_wb",
        ngram_range=(1, 1) if tokenizer else (5, 5),
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
    n_neighbors = min(n_neighbors, len(corpus))
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
    tokenizer: Optional[Callable] = None,
    n_neighbors: int = 100,
):
    table_paths = sorted(Path(data_dir).glob("[1-2]*.csv"))
    dfs = [pd.read_csv(p, index_col=index_col) for p in table_paths]

    matches_path = Path(data_dir) / "matches.csv"
    matches = set(pd.read_csv(matches_path).itertuples(index=False, name=None))

    queries = convert_df_to_corpus(dfs[0])
    corpus = convert_df_to_corpus(dfs[-1])
    vectorizer = fit_vectorizer(corpus, tokenizer=tokenizer)
    index = build_index(corpus, vectorizer, n_neighbors=n_neighbors)
    indices_list = knn_join(queries, vectorizer, index)

    candidates = get_candidates(dfs, indices_list)
    metrics = evaluate(candidates, matches)

    print(metrics)
    return metrics


if __name__ == "__main__":
    CLI(sparse_join)
