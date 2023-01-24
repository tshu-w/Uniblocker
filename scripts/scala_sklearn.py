import re
import sys
from pathlib import Path

import numpy as np

np.int = int
np.float = float
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer

sys.path.append(str(Path(__file__).parents[1]))

from src.baselines.sparse_join import sparse_join


def main():
    data_dirs = [
        d
        for d in Path("./data/blocking").iterdir()
        if d.name in ["songs", "citeseer-dblp"]
    ]
    sizes = [100, 1000, 10000, 100000, 1000000]
    # for d in data_dirs:
    #     for f in Path(d).glob("[1-2]*.csv"):
    #         df = pd.read_csv(f, low_memory=False)
    #         for i in sizes:
    #             sub_f = d / f"{f.stem}_{i}.csv"
    #             sub_df = df.head(i)
    #             sub_df.to_csv(sub_f, index=False)

    for d in data_dirs:
        print(d.name)
        for i in sizes:
            print(i)
            tokenizers = [
                re.compile(r"(?u)\b\w\w+\b").findall,
                WhitespaceTokenizer().tokenize,
                QgramTokenizer(qval=4).tokenize,
                QgramTokenizer(qval=5).tokenize,
                QgramTokenizer(qval=6).tokenize,
            ]
            for tokenizer in tokenizers:
                sparse_join(data_dir=d, size=str(i), tokenizer=tokenizer)


if __name__ == "__main__":
    main()
