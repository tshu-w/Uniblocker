from functools import partial
from typing import Literal, Optional

import numpy as np
import pytorch_lightning as pl
from datasets import Dataset
from pytorch_lightning import Callback
from pytorch_lightning.utilities import move_data_to_device
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.datamodules.blocking import mapping2tuple
from src.utils import chunks, evaluate, get_candidates


class EmptyIterDataset(IterableDataset):
    def __iter__(self):
        return iter([])


def empty_fun(*args, **kwargs):
    pass


def empty_dataloader(*args, **kwargs):
    return DataLoader(EmptyIterDataset())


class Evaluator(Callback):
    def __init__(
        self,
        n_neighbors: int = 100,
        direction: Literal["forward", "reversed", "both"] = "forward",
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.direction = direction

    def setup(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        datamodule = trainer.datamodule or module
        module.validation_step = module.test_step = empty_fun
        datamodule.val_dataloader = empty_dataloader
        datamodule.test_dataloader = empty_dataloader

    def evaluate(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        datamodule = trainer.datamodule or module
        index_col = datamodule.hparams.index_col
        knn_join = partial(Evaluator.knn_join, n_neighbors=self.n_neighbors)

        datasets = [Dataset.from_pandas(ds.df) for ds in datamodule.datasets]
        datasets = Evaluator.build_index(
            datasets,
            module=module,
            datamodule=datamodule,
            index_col=index_col,
        )

        if len(datasets) == 1:
            indices_list = [knn_join(corpus=datasets[0], index=datasets[0])]
        else:
            indices_list = [
                knn_join(corpus=datasets[0], index=datasets[1]),
                knn_join(corpus=datasets[1], index=datasets[0]),
            ]

        dfs = [d.to_pandas().set_index(index_col) for d in datasets]

        candidates = get_candidates(
            dfs, indices_list, n_neighbors=self.n_neighbors, direction=self.direction
        )
        matches = datamodule.matches
        results = evaluate(candidates, matches)

        if trainer.validating:
            module.log_dict({f"val/{k}": v for k, v in results.items()}, sync_dist=True)
        else:
            module.log_dict(results, sync_dist=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, module: pl.LightningModule
    ) -> None:
        self.evaluate(trainer, module)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, module: pl.LightningModule
    ) -> None:
        self.evaluate(trainer, module)

    @staticmethod
    def build_index(
        datasets: list[Dataset],
        module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        index_col: str,
    ) -> list[Dataset]:
        def encode(batch: dict[list]):
            collate_fn = getattr(module, "collate_fn", default_collate)

            batch: list[dict] = [dict(zip(batch, t)) for t in zip(*batch.values())]
            batch = [mapping2tuple(r, index_col) for r in batch]
            batch = move_data_to_device(collate_fn(batch), module.device)

            embeddings = module(batch).detach().to("cpu").numpy()
            embeddings = normalize(embeddings).astype(np.float32)

            return {"embeddings": embeddings}

        batch_size = datamodule.hparams.batch_size

        for i, dataset in enumerate(datasets):
            datasets[i] = dataset.map(
                encode,
                batched=True,
                batch_size=batch_size,
                load_from_cache_file=False,
            )
            datasets[i].add_faiss_index(column="embeddings", faiss_verbose=True)
            datasets[i].set_format("numpy")

        return datasets

    @staticmethod
    def knn_join(
        n_neighbors: int,
        *,
        corpus: Dataset,
        index: Dataset,
        chunk_size: int = 64,
    ) -> list[list[int]]:
        indices_list = []
        for record in tqdm(list(chunks(corpus, chunk_size))):
            queries = record["embeddings"]
            scores, indices = index.search_batch(
                index_name="embeddings", queries=queries, k=n_neighbors
            )
            assert np.all(np.diff(scores) >= 0)
            indices_list.append(indices)

        indices = np.concatenate(indices_list).tolist()

        return indices
