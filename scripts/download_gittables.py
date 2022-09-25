import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import requests
from tqdm import tqdm

rid = "6517052"
access_token = os.getenv("ZENODO_ACCESS_TOKEN")
r = requests.get(
    f"https://zenodo.org/api/records/{rid}?access_token={access_token}",
)
dir = Path(f"./data/gittables/raw_{rid}")
dir.mkdir(parents=True, exist_ok=True)

for f in tqdm(r.json()["files"]):
    filename = dir / f["key"]
    if not filename.exists():
        urlretrieve(f["links"]["self"], filename)

    if not (dir / filename.stem).exists():
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(dir / filename.stem)
