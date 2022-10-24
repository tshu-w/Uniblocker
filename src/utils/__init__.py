from typing import Iterable, Iterator, Optional, Union

import pandas as pd

from .evaluation import evaluate, get_candidates


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def mapping2tuple(
    record: Union[dict, pd.Series], index_col: Optional[str] = None
) -> list[tuple]:
    record = list(filter(lambda x: x[0] != index_col, record.items()))
    record = list(
        (str(t[0]).casefold(), str(t[1]).casefold()) for t in record if t[1] is not None
    )
    return record
