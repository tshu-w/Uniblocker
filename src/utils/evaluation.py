from typing import Any

import pandas as pd
from sklearn.metrics import auc


def get_candidates(
    dfs: list[pd.DataFrame],
    indices_list: list[list[int]],
) -> list[set[tuple[Any, Any]]]:
    candidates = []
    flags = set()  # Comparison Propagation
    for i in range(len(indices_list[0])):
        cands = set()
        for j in range(len(indices_list)):
            ind1 = j
            ind2 = indices_list[j][i]
            if len(dfs) == 1 and ind1 > ind2:
                ind1, ind2 = ind2, ind1

            id1 = dfs[0].index[ind1]
            id2 = dfs[len(dfs) - 1].index[ind2]
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
    cands = set()
    precisions, recalls = [1], [0]

    for i in range(len(candidates)):
        cands = cands | candidates[i]

        tp = len(cands & matches)
        precision = tp / len(cands)
        recall = tp / len(matches)

        precisions.append(precision)
        recalls.append(recall)

    k = 0
    for i in range(len(candidates) + 1):
        precision = precisions[i]
        recall = recalls[i]
        k = i
        if recall > threshold:
            break

    average_precision = auc(recalls, precisions)

    return {
        "AP": average_precision,
        "PC": recall,
        "PQ": precision,
        "F1": 2 * (precision * recall) / (precision + recall),
        "K": float(k),
    }
