from typing import Iterable, Iterator, Optional

from .evaluation import evaluate, get_candidates
from .table_detector import check_table


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def dict2tuples(record: dict, index_col: Optional[str] = None) -> list[tuple]:
    record = list(filter(lambda x: x[0] != index_col, record.items()))
    record = list((str(t[0]).casefold(), str(t[1]).casefold()) for t in record if t[1])
    return record
