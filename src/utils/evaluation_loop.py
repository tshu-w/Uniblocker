from typing import Any, TypeVar

import numpy as np
from pytorch_lightning.loops import Loop
from pytorch_lightning.trainer.connectors.logger_connector.result import (
    _OUT_DICT,
    ResultCollection,
)
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

T = TypeVar("T")  # the output type of `run`


class EvaluationLoop(Loop):
    def __init__(self, k: int = 50) -> None:
        super().__init__()

        self.k = k
        self.outputs: list[_OUT_DICT] = []
        self._results = ResultCollection(training=False)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        datamodule = self.trainer.datamodule

        datasets = datamodule.datasets
        batch_size = datamodule.hparams.batch_size
        golden_pairs = datamodule.golden_pairs

        for i, dataset in enumerate(datamodule.datasets):
            datasets[i] = dataset.map(
                self.encode,
                batched=True,
                batch_size=batch_size,
            )
            datasets[i].add_faiss_index(column="embeddings")
            datasets[i].reset_format()

        candidate_pairs = set()
        for record in tqdm(datasets[0]):
            query = np.array(record["embeddings"], dtype=np.float32)
            _scores, examples = datasets[1].get_nearest_examples(
                index_name="embeddings", query=query, k=self.k
            )
            candidate_pairs |= set(zip([record["id"]] * self.k, examples["id"]))

        results = {
            "recall": len(candidate_pairs & golden_pairs) / len(golden_pairs),
            "cssr": len(candidate_pairs) / (len(datasets[0]) + len(datasets[1])),
        }
        self.trainer.logger_connector._logged_metrics.update(results)
        self.outputs = [results]

    def on_run_end(self) -> T:
        outputs, self.outputs = self.outputs, []  # free memory
        return outputs

    @property
    def done(self) -> bool:
        return self.outputs

    def reset(self) -> None:
        self._logged_outputs = []

    def encode(self, batch: dict[list]):
        model = self.trainer.model
        collate_fn = getattr(self.trainer.model, "collate_fn", default_collate)

        batch: list[dict] = [dict(zip(batch, t)) for t in zip(*batch.values())]
        batch = move_data_to_device(collate_fn(batch), model.device)

        embeddings = model(batch).detach().to("cpu").numpy()

        return {"embeddings": embeddings}

    def _reload_evaluation_dataloaders(self):
        pass
