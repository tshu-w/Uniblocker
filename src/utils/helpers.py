from typing import Any, Iterable, Iterator


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    size = lst.shape[0] if hasattr(lst, "shape") else len(lst)
    for i in range(0, size, n):
        yield lst[i : i + n]


def record2str(record: dict) -> str:
    return " ".join(str(v).casefold() for v in record.values() if v)


def evaluate(
    candidates: list[set[tuple[Any, Any]]],
    matches: set[tuple[Any, Any]],
    *,
    threshold: float = 0.9,
) -> dict:
    cands = set()
    precisions, recalls = [1], [0]

    average_precision = 0
    for i in range(len(candidates)):
        cands = cands | candidates[i]

        tp = len(cands & matches)
        precision = tp / len(cands)
        recall = tp / len(matches)
        average_precision += precision * (recall - recalls[-1])

        precisions.append(precision)
        recalls.append(recall)

    k = 0
    for i in range(len(candidates) + 1):
        precision = precisions[i]
        recall = recalls[i]
        k = i
        if recall > threshold:
            break

    return {
        "AP": average_precision,
        "PC": recall,
        "PQ": precision,
        "F1": 2 * (precision * recall) / (precision + recall),
        "K": float(k),
    }
