from typing import Any, Iterable, Iterator, Literal, Optional

import pandas as pd
from sklearn.metrics import auc

from . import loggers, shtab


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_candidates(
    dfs: list[pd.DataFrame],
    indices_list: list[list[list]],
    *,
    n_neighbors: int = 100,
    direction: Optional[Literal["forward", "reversed", "both"]] = None,
) -> list[set[tuple[Any, Any]]]:
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
                pair = id1, id2
                if id1 != id2 and pair not in flags:
                    cands.add(pair)
                    flags.add(pair)

        if direction == "reversed":
            for j in range(len(dfs[1])):
                ind1 = indices_list[1][j][i]
                ind2 = j
                id1 = dfs[0].index[ind1]
                id2 = dfs[1].index[ind2]
                pair = id1, id2
                if id1 != id2 and pair not in flags:
                    cands.add(pair)
                    flags.add(pair)

        candidates.append(cands)

    return candidates


def evaluate(
    candidates: list[set[tuple[Any, Any]]],
    matches: set[tuple[Any, Any]],
    *,
    threshold: float = 0.9,
) -> dict:
    # TODO: CSSR len(tp) / len(data)
    # TODO: MRR (mean reciprocal rank)
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
