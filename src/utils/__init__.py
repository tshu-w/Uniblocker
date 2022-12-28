import os
import tempfile
from multiprocessing import connection, util
from multiprocessing.connection import _mmap_counter

from .evaluation import evaluate, get_candidates
from .helpers import chunks, dict2tuples, tuples2str
from .table_detector import check_table


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


# HACK: https://github.com/SeldonIO/seldon-core/issues/3720
connection.arbitrary_address = arbitrary_address
