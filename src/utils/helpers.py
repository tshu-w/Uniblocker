from typing import Iterable, Iterator


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def record2str(record: dict) -> str:
    return " ".join(str(v).casefold() for v in record.values() if v)
