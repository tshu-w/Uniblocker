import os
import sys
from pathlib import Path

os.environ[
    "CLASSPATH"
] = f"{str(Path(__file__).parents[1])}/src/vendor/jedai-core-3.2.1-jar-with-dependencies.jar"
sys.path.append(str(Path(__file__).parents[1]))

from src.baselines.blocking_workflows import blocking_workflows


def main():
    data_dirs = [
        d
        for d in Path("./data/blocking").iterdir()
        if d.name in ["songs", "citeseer-dblp"]
    ]
    sizes = [100, 1000, 10000, 100000, 1000000]

    for d in data_dirs:
        print(d.name)
        for i in sizes:
            print(i)
            blocking_workflows(data_dir=d, size=str(i))


if __name__ == "__main__":
    main()
