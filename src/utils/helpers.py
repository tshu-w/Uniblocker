from typing import Iterable, Iterator


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def dict2tuples(record: dict, ignored_cols: list[str] = ["id"]) -> list[tuple]:
    record = list(filter(lambda x: x[0] not in ignored_cols, record.items()))
    record = list((str(t[0]).casefold(), str(t[1]).casefold()) for t in record if t[1])
    return record


def tuples2str(record: list[tuple]) -> str:
    return " ".join([t[1] for t in record])