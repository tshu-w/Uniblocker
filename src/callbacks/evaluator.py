from functools import partial
from typing import Literal, Optional

import numpy as np
import py_stringmatching as sm
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import Dataset
from pytorch_lightning import Callback
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.datamodules.blocking import dict2tuples
from src.utils import chunks, evaluate, get_candidates


class EmptyIterDataset(IterableDataset):
    def __iter__(self):
        return iter([])


def empty_fun(*args, **kwargs):
    pass


def empty_dataloader(*args, **kwargs):
    return DataLoader(EmptyIterDataset())


def sparse_similarity(
    s1,
    s2,
    tokenizer=sm.tokenizer.whitespace_tokenizer.WhitespaceTokenizer(),
    similarity=sm.similarity_measure.cosine.Cosine(),
):
    t1 = tokenizer.tokenize(s1)
    t2 = tokenizer.tokenize(s2)
    return similarity.get_sim_score(t1, t2)


class Evaluator(Callback):
    def __init__(
        self,
        n_neighbors: int = 100,
        direction: Literal["forward", "reversed", "both"] = "forward",
        ensemble: bool = False,
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.direction = direction
        self.ensemble = ensemble

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
        knn_join = partial(
            Evaluator.knn_join,
            n_neighbors=self.n_neighbors,
            ensemble=self.ensemble,
        )

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
        @torch.no_grad()
        def encode(batch: dict[list]):
            collate_fn = getattr(module, "collate_fn", default_collate)

            batch: list[dict] = [dict(zip(batch, t)) for t in zip(*batch.values())]
            batch = [dict2tuples(r, ignored_cols=[index_col]) for r in batch]
            texts = [" ".join([t[1] for t in l]) for l in batch]
            batch = move_data_to_device(collate_fn(batch), module.device)

            embeddings = F.normalize(module(batch).detach()).to("cpu").numpy()

            return {
                "text": texts,
                "embeddings": embeddings.astype(np.float32),
            }

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
        ensemble: bool = True,
    ) -> list[list[int]]:
        indices_list = []
        for record in tqdm(list(chunks(corpus, chunk_size))):
            queries = record["embeddings"]
            scores, indices = index.search_batch(
                index_name="embeddings", queries=queries, k=n_neighbors
            )
            if ensemble:
                query_texts = record["text"]
                candidates_texts = [index[idx]["text"] for idx in indices]
                for i, s1 in enumerate(query_texts):
                    for j, s2 in enumerate(candidates_texts[i]):
                        scores[i, j] = (
                            scores[i, j] * 0.5 + (1 - sparse_similarity(s1, s2)) * 0.5
                        )

                scores_ind = scores.argsort()
                scores = np.take_along_axis(scores, scores_ind, axis=-1)
                indices = np.take_along_axis(indices, scores_ind, axis=-1)

            assert np.all(np.diff(scores) >= 0)
            indices_list.append(indices)

        indices = np.concatenate(indices_list).tolist()

        return indices
