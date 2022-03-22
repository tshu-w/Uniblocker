import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("../data/gittables_raw/")
DATA_DIR = Path("../data/gittables/")

for path in RAW_DIR.iterdir():
    if not path.is_dir():
        continue

    with open(DATA_DIR / f"{path.name}.jsonl", "w") as f:
        for file in tqdm(path.rglob("*.parquet")):
            try:
                df = pd.read_parquet(file)
                df.drop_duplicates(inplace=True)
            except Exception as e:
                print(file)

            filename = str(file.relative_to(DATA_DIR))
            df.columns = df.columns.str.replace("\ufeff", "")
            tuples = df.to_dict("records")
            for i, t in enumerate(tuples):
                tuples[i] = json.dumps(
                    {
                        "_file": filename,
                        "_idx": i,
                        "tuple": json.dumps(t, ensure_ascii=False),
                    }
                )

            f.writelines(t + "\n" for t in tuples)
