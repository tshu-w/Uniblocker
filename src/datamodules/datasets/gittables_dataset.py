from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from src.utils.table_detector import check_table


class GitTablesDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        data_files: Optional[list[str]] = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.data_files = data_files
        if data_files is not None:
            self.data_files = [data_dir / f for f in data_files]

    def __iter__(self) -> dict:
        worker_info = torch.utils.data.get_worker_info()
        for i, f in enumerate(self.data_files or self.data_dir.rglob("*.parquet")):
            if worker_info is None or i % worker_info.num_workers == worker_info.id:
                try:
                    df = pd.read_parquet(f)
                    df.columns = df.columns.str.replace("\ufeff", "")
                    df.drop_duplicates(inplace=True)
                except Exception:
                    continue

                if check_table(df):
                    df = df.fillna("").astype(str)
                    yield from df.to_dict("records")
