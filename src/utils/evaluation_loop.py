from typing import Any

import numpy as np
from datasets import Dataset
from pytorch_lightning.loops.dataloader import evaluation_loop
from pytorch_lightning.utilities import move_data_to_device
from sklearn.preprocessing import normalize
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.utils import chunks, evaluate, get_candidates


class EvaluationLoop(evaluation_loop.EvaluationLoop):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs the ``_on_evaluation_model_eval``, ``_on_evaluation_start`` and ``_on_evaluation_epoch_start``
        hooks."""
        # hook
        self._on_evaluation_model_eval()
        self.trainer.lightning_module.zero_grad()
        self._on_evaluation_start()
        self._on_evaluation_epoch_start()

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        ...

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs evaluation on one single dataloader."""
        # required for logging
        self.trainer.lightning_module._current_fx_name = "validation_step"

        self.build_index()
        datasets = [d.with_format("numpy") for d in self.trainer.datamodule.datasets]
        if len(datasets) == 1:
            indices_list = [self.knn_join(corpus=datasets[0], index=datasets[0])]
        else:
            indices_list = [
                self.knn_join(corpus=datasets[0], index=datasets[1]),
                self.knn_join(corpus=datasets[1], index=datasets[0]),
            ]

        index_col = self.trainer.datamodule.hparams.index_col
        dfs = [d.to_pandas().set_index(index_col) for d in datasets]

        n_neighbors = self.trainer.datamodule.hparams.n_neighbors
        direction = self.trainer.datamodule.hparams.direction
        candidates = get_candidates(
            dfs, indices_list, n_neighbors=n_neighbors, direction=direction
        )
        matches = self.trainer.datamodule.matches
        results = evaluate(candidates, matches)

        assert self.trainer.datamodule.datasets[0].format["type"] == "torch"

        self.trainer.model.log_dict(results)

        # store batch level output per dataloader
        self._outputs.append(results)

        if not self.trainer.sanity_checking:
            # indicate the loop has run
            self._has_run = True

    @property
    def skip(self) -> bool:
        return False

    @property
    def done(self) -> bool:
        return self._outputs

    def encode(self, batch: dict[list]):
        model = self.trainer.model
        collate_fn = getattr(self.trainer.model, "collate_fn", default_collate)

        batch: list[dict] = [dict(zip(batch, t)) for t in zip(*batch.values())]
        batch = move_data_to_device(collate_fn(batch), model.device)

        embeddings = model(batch).detach().to("cpu").numpy()
        embeddings = normalize(embeddings).astype(np.float32)

        return {"embeddings": embeddings}

    def build_index(self):
        datamodule = self.trainer.datamodule
        datasets = datamodule.datasets
        batch_size = datamodule.hparams.batch_size

        for i, dataset in enumerate(datasets):
            datasets[i] = dataset.map(
                self.encode,
                batched=True,
                batch_size=batch_size,
                load_from_cache_file=False,
            )
            datasets[i].add_faiss_index(column="embeddings", faiss_verbose=True)

    def knn_join(
        self,
        *,
        corpus: Dataset,
        index: Dataset,
        chunk_size: int = 64,
    ) -> list[list[int]]:
        indices_list = []
        n_neighbors = self.trainer.datamodule.hparams.n_neighbors
        for record in tqdm(list(chunks(corpus, chunk_size))):
            queries = record["embeddings"]
            _scores, indices = index.search_batch(
                index_name="embeddings", queries=queries, k=n_neighbors
            )
            indices_list.append(indices)

        indices = np.concatenate(indices_list).tolist()

        return indices
