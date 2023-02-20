import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate

from src.utils import evaluate
from src.utils.nns_blocker import FaissIndexer, NeuralConverter, NNSBlocker


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
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors

    def evaluate(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        datamodule = trainer.datamodule or module

        dfs = [ds.df for ds in datamodule.datasets]
        collate_fn = getattr(module, "collate_fn", default_collate)
        converter = NeuralConverter(module, collate_fn, module.device)

        # import nmslib
        # from src.utils.nns_blocker import NMSLIBIndexer
        # indexer = NMSLIBIndexer(
        #     init_kwargs={
        #         "method": "hnsw",
        #         "space": "cosinesimil",
        #         "data_type": nmslib.DataType.DENSE_VECTOR,
        #     },
        #     index_params={"M": 30, "indexThreadQty": 12, "efConstruction": 1000},
        #     query_params={},
        #     threads=12,
        # )
        indexer = FaissIndexer(
            index_params={
                "save_on_disk": False,
                "min_nearest_neighbors_to_retrieve": self.n_neighbors,
                "index_key": "Flat",
                "verbose": 30,  # logging.WARNING
            },
        )
        blocker = NNSBlocker(dfs, converter, indexer)
        candidates = blocker(k=self.n_neighbors)

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
