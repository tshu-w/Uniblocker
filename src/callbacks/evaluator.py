import copy
from functools import partial
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import move_data_to_device
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.utils import chunks, evaluate, get_candidates


class EmptyIterDataset(IterableDataset):
    def __iter__(self):
        return iter([])


def empty_fun(*args, **kwargs):
    pass


def empty_dataloader(*args, **kwargs):
    return DataLoader(EmptyIterDataset())


class Evaluator(Callback):
    def __init__(self) -> None:
        super().__init__()

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
        build_index = partial(
            Evaluator.build_index, module=module, datamodule=datamodule
        )
        knn_join = partial(Evaluator.knn_join, datamodule=datamodule)

        datasets = [copy.deepcopy(d) for d in datamodule.datasets]
        datasets = build_index(datasets)

        if len(datasets) == 1:
            indices_list = [knn_join(corpus=datasets[0], index=datasets[0])]
        else:
            indices_list = [
                knn_join(corpus=datasets[0], index=datasets[1]),
                knn_join(corpus=datasets[1], index=datasets[0]),
            ]

        index_col = datamodule.hparams.index_col
        dfs = [d.to_pandas().set_index(index_col) for d in datasets]

        n_neighbors = datamodule.hparams.n_neighbors
        direction = datamodule.hparams.direction
        candidates = get_candidates(
            dfs, indices_list, n_neighbors=n_neighbors, direction=direction
        )
        matches = datamodule.matches
        results = evaluate(candidates, matches)

        if trainer.validating:
            module.log_dict({f"val/{k}": v for k, v in results.items()}, sync_dist=True)
        else:
            module.log_dict(results, sync_dist=True)

        assert datamodule.datasets[0].format["type"] == "torch"

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
    ) -> list[Dataset]:
        def encode(batch: dict[list]):
            collate_fn = getattr(module, "collate_fn", default_collate)

            batch: list[dict] = [dict(zip(batch, t)) for t in zip(*batch.values())]
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
        datamodule,
        *,
        corpus: Dataset,
        index: Dataset,
        chunk_size: int = 64,
    ) -> list[list[int]]:
        indices_list = []
        n_neighbors = datamodule.hparams.n_neighbors
        for record in tqdm(list(chunks(corpus, chunk_size))):
            queries = record["embeddings"]
            scores, indices = index.search_batch(
                index_name="embeddings", queries=queries, k=n_neighbors
            )
            assert np.all(np.diff(scores) >= 0)
            indices_list.append(indices)

        indices = np.concatenate(indices_list).tolist()

        return indices
