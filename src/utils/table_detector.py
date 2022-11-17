import re
from statistics import variance
from typing import Iterable, Literal

import pandas as pd
from dateutil.parser import parse as dateutil_parse
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def is_url(
    s: str,
    regex=re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    ),  # taken from: django.core.validators
) -> bool:
    return s and regex.search(s) is not None


def is_datetime(s: str) -> bool:
    try:
        dateutil_parse(s, ignoretz=True)
        return True
    except Exception:
        return False


def most(iterable: Iterable, threshold: float = 0.8) -> bool:
    tlt = cnt = 0
    for element in iterable:
        tlt += 1
        if element:
            cnt += 1
    return cnt >= tlt * threshold


def most_ascii(s: str) -> bool:
    return most([ch.isascii() for ch in s if ch.isalnum()])


def is_english_table(table: pd.DataFrame) -> bool:
    return most(
        most_ascii(s)
        for s in table.head(n=10).astype(str).apply(lambda row: " ".join(row), axis=1)
    )


def get_column_type(
    column: pd.Series,
) -> Literal["identifier", "numeric", "url", "date", "category", "entity", "text"]:
    """
    Heuristic rules to detect column type of pd.DataFrame.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7573692
    """
    if "id" in column.name.lower():
        return "identifier"

    column_wo_na = column.dropna()

    if is_numeric_dtype(column_wo_na) or is_bool_dtype(column_wo_na):
        return "numeric"

    str_column_wo_na = column_wo_na.astype(str)

    if most(str_column_wo_na.apply(is_url), 0.5):
        return "url"

    if most(str_column_wo_na.apply(is_datetime)):
        return "date"

    lens = str_column_wo_na.str.split().apply(len)
    if len(lens) < 2 or variance(lens) == 0:
        if len(column.drop_duplicates()) / len(column) > 0.8:
            return "identifier"
        else:
            return "category"
    else:
        if sum(lens) / len(column) < 20:
            return "entity"
        else:
            return "text"


def is_entity_table(table: pd.DataFrame) -> bool:
    """
    Heuristic rules to skip not latin tables and check if a table
    is an entity tables (not tabular, annotation, log and others).
    """
    if any(table.columns.duplicated()):
        return False

    column_types = [get_column_type(table[c]) for c in table.columns]
    return (
        most((lambda t: t not in ["numeric", "url", "date"] for t in column_types), 0.5)
        and len(column_types)
        and column_types[0] != "date"
    )


def check_table(table: pd.DataFrame) -> bool:
    """
    Whether table should is valid.
    """
    return len(table) and is_english_table(table) and is_entity_table(table)


if __name__ == "__main__":
    data_files = "./data/gittables".rglob("*.parquet")

    for f in data_files:
        try:
            df = pd.read_parquet(f)
            df.columns = df.columns.str.replace("\ufeff", "")
            df.drop_duplicates(inplace=True)
        except Exception:
            continue

        if not check_table(df) and len(df):
            print(df.head())
