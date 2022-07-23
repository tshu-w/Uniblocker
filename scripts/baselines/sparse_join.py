from collections import namedtuple
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from jsonargparse import CLI
from rich import print
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import auc
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

Pair = namedtuple("Pair", ["id1", "id2"])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def evaluate(
    candidates: list[set[Pair]],
    matches: set[Pair],
    *,
    threshold: float = 0.9,
) -> dict:
    # AP
    cands = set()
    precisions, recalls = [], []

    for i in range(len(candidates)):
        cands = cands | candidates[i]

        tp = len(cands & matches)
        precision = tp / len(cands)
        recall = tp / len(matches)

        precisions.append(precision)
        recalls.append(recall)

    k = -1
    for i in range(len(candidates)):
        precision = precisions[i]
        recall = recalls[i]
        if recall > threshold:
            k = i
            break

    average_precision = auc(recalls, precisions)

    return {
        "AP": average_precision,
        "PC": recall,
        "PQ": precision,
        "F1": 2 * (precision * recall) / (precision + recall),
        "k": k,
    }


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
    index.fit(vectorizer.fit_transform(corpus))

    return index


def knn_join(
    corpus: list[str],
    vectorizer: CountVectorizer,
    index: NearestNeighbors,
    *,
    n_neighbors: Optional[int] = None,
    chunk_size: int = 128,
) -> tuple[np.array, np.array]:
    indices_list = []

    for texts in tqdm(chunks(corpus, chunk_size)):
        vectors = vectorizer.transform(texts)
        _distances, indices = index.kneighbors(vectors, n_neighbors=n_neighbors)
        indices_list.append(indices)

    indices = np.concatenate(indices_list).tolist()

    return indices


def get_candidates(
    dfs: list[pd.DataFrame],
    indices_list: list[list[list]],
    *,
    n_neighbors: int = 100,
    direction: Optional[Literal["forward", "reversed", "both"]] = None,
) -> list[set[Pair]]:
    candidates = []
    flags = set()  # Comparison Propagation
    for i in range(n_neighbors):
        cands = set()
        if direction != "reversed":
            for j in range(len(dfs[0])):
                ind1 = j
                ind2 = indices_list[0][j][i]
                if len(dfs) == 1 and ind1 > ind2:
                    ind1, ind2 = ind2, ind1

                id1 = dfs[0].index[ind1]
                id2 = dfs[len(dfs) - 1].index[ind2]
                pair = Pair(id1, id2)
                if id1 != id2 and pair not in flags:
                    cands.add(pair)
                    flags.add(pair)

        if direction == "reversed":
            for j in range(len(dfs[1])):
                ind1 = indices_list[1][j][i]
                ind2 = j
                id1 = dfs[0].index[ind1]
                id2 = dfs[1].index[ind2]
                pair = Pair(id1, id2)
                if id1 != id2 and pair not in flags:
                    cands.add(pair)
                    flags.add(pair)

        candidates.append(cands)

    return candidates


def sparse_join(
    data_path: str = "./data/blocking/cora",
    index_col: str = "id",
    n_neighbors: int = 100,
    direction: Optional[Literal["forward", "reversed", "both"]] = None,
):
    paths = sorted(Path(data_path).glob("[1-2]*.csv"))
    assert len(paths) == 1 or direction is not None
    assert len(paths) == 2 or direction is None

    dfs = [pd.read_csv(p, index_col=index_col) for p in paths]
    corpuses = [convert_df_to_corpus(df) for df in dfs]

    matches_path = str(Path(data_path) / "matches.csv")
    matches = set(pd.read_csv(matches_path).itertuples(index=False, name="Pair"))

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
    results = evaluate(candidates, matches)

    print(results)
    return results


if __name__ == "__main__":
    CLI(sparse_join)
