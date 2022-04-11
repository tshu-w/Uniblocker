import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("../data/gittables/")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for path in RAW_DIR.iterdir():
    if not path.is_dir():
        continue

    with open(PROCESSED_DIR / f"{path.name}.jsonl", "w") as f:
        for file in tqdm(path.rglob("*.parquet")):
            try:
                df = pd.read_parquet(file)
                df.drop_duplicates(inplace=True)
            except Exception as e:
                print(file)

            filename = str(file.relative_to(RAW_DIR))
            df.columns = df.columns.str.replace("\ufeff", "")
            records = df.to_dict("records")
            for i, r in enumerate(records):
                records[i] = json.dumps(
                    {
                        "_file": filename,
                        "_idx": i,
                        "record": json.dumps(r, ensure_ascii=False),
                    }
                )

            f.writelines(r + "\n" for r in records)
