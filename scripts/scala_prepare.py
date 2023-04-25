import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import pandas as pd


def main():
    data_dirs = [
        d
        for d in Path("./data/blocking").iterdir()
        if d.name in ["songs", "citeseer-dblp"]
    ]
    sizes = [100, 1000, 10000, 100000, 1000000]
    for d in data_dirs:
        for f in Path(d).glob("[1-2]*.csv"):
            df = pd.read_csv(f, low_memory=False)
            for i in sizes:
                sub_f = d / f"{f.stem}_{i}.csv"
                sub_df = df.head(i)
                sub_df.to_csv(sub_f, index=False)


if __name__ == "__main__":
    main()
