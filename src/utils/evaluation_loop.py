from typing import Any

import numpy as np
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


class EvaluationLoop(EvaluationLoop):
    def __init__(self, k: int = 50) -> None:
        super().__init__()

        self.k = k

    def advance(self, *args: Any, **kwargs: Any) -> None:
        # required for logging
        self.trainer.lightning_module._current_fx_name = "validation_step"
        self.set_format()
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

        self.set_format()
        results = {
            "recall": len(candidate_pairs & golden_pairs) / len(golden_pairs),
            "cssr": len(candidate_pairs) / (len(datasets[0]) + len(datasets[1])),
        }
        self.trainer.model.log_dict(results)

        self.outputs = [results]

    @property
    def num_dataloaders(self):
        return 1

    @property
    def skip(self) -> bool:
        return False

    @property
    def done(self) -> bool:
        return self.outputs

    def encode(self, batch: dict[list]):
        model = self.trainer.model
        collate_fn = getattr(self.trainer.model, "collate_fn", default_collate)

        batch: list[dict] = [dict(zip(batch, t)) for t in zip(*batch.values())]
        batch = move_data_to_device(collate_fn(batch), model.device)

        embeddings = model(batch).detach().to("cpu").numpy()

        return {"embeddings": embeddings}

    def set_format(self):
        feature_columns = getattr(self.trainer.model, "feature_columns", None)
        for dataset in self.trainer.datamodule.datasets:
            dataset.set_format(type="torch", columns=feature_columns)
