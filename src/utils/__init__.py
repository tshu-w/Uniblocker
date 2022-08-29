from typing import Any, Iterable, Iterator, Literal, Optional

from . import loggers, shtab, training_epoch_loop
from .evaluation import evaluate, get_candidates


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
