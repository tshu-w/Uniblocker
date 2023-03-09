import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.baselines.faiss_join import faiss_join
from src.models import UniBlocker


def main():
    data_dirs = [
        d
        for d in Path("./data/blocking").iterdir()
        if d.name in ["songs", "citeseer-dblp"]
    ]
    sizes = [100, 1000, 10000, 100000, 1000000]

    model = UniBlocker(
        model_name_or_path="roberta-base",
        max_length=256,
    )
    for d in data_dirs:
        print(d.name)
        for i in sizes:
            print(i)
            faiss_join(model=model, data_dir=d, size=str(i))


if __name__ == "__main__":
    main()
