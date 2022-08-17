from collections import namedtuple

from sklearn.metrics import auc

from . import loggers, shtab

RecordPair = namedtuple("RecordPair", ["id1", "id2"])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def evaluate(
    candidates: list[set[RecordPair]],
    matches: set[RecordPair],
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
