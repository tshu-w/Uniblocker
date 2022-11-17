from typing import Iterable, Iterator, Optional

from .evaluation import evaluate, get_candidates
from .table_detector import check_table


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def dict2tuples(record: dict, ignored_cols: list[str] = ["id"]) -> list[tuple]:
    record = list(filter(lambda x: x[0] not in ignored_cols, record.items()))
    record = list((str(t[0]).casefold(), str(t[1]).casefold()) for t in record if t[1])
    return record


# HACK: https://github.com/SeldonIO/seldon-core/issues/3720
import os
import tempfile
from multiprocessing import connection, util
from multiprocessing.connection import _mmap_counter


def arbitrary_address(family):
    """
    Return an arbitrary free address for the given family
    """
    if family == "AF_INET":
        return ("localhost", 0)
    elif family == "AF_UNIX":
        return tempfile.mktemp(prefix="listener-", dir=util.get_temp_dir())
    elif family == "AF_PIPE":
        return tempfile.mktemp(
            prefix=r"\\.\pipe\pyc-%d-%d-" % (os.getpid(), next(_mmap_counter)), dir=""
        )
    else:
        raise ValueError("unrecognized family")


connection.arbitrary_address = arbitrary_address
