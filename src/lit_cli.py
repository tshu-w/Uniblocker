#!/usr/bin/env python

import json
import logging
from collections import ChainMap
from datetime import datetime
from pathlib import Path
from typing import Any

import shtab
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI
from rich import print


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        for arg in ["batch_size", "num_labels", "task_name"]:
            parser.link_arguments(
                f"data.init_args.{arg}",
                f"model.init_args.{arg}",
                apply_on="instantiate",
            )

    def modify_logger(self, logger: LightningLoggerBase, exp_name: str, version: str):
        if exp_name and hasattr(logger, "_name"):
            logger._name = exp_name

        if version and hasattr(logger, "_version"):
            logger._version = version

    def before_run(self):
        model_name = type(self.model).__name__
        datamodule_name = type(self.datamodule).__name__ if self.datamodule else ""
        exp_name = "_".join(filter(None, [model_name, datamodule_name]))

        model_version = (
            self.model.get_version() if hasattr(self.model, "get_version") else ""
        )
        datamodule_version = (
            self.datamodule.get_version()
            if hasattr(self.datamodule, "get_version")
            else ""
        )
        seed = str(self.seed_everything_default)
        timestramp = datetime.now().strftime("%m%d-%H%M%S")
        version = "_".join(
            filter(None, [model_version, datamodule_version, seed, timestramp])
        )
        log_dir = (
            f"{self.trainer.default_root_dir}/{exp_name.lower()}/{version.lower()}"
        )

        print(f"Experiment: [bold]{exp_name}[/bold]")
        print(f"Version:    [bold]{version}[/bold]")
        print(f"Log Dir:    [bold]{log_dir}[/bold]")

        if isinstance(self.trainer.logger, LoggerCollection):
            for logger in self.trainer.logger:
                self.modify_logger(logger, exp_name.lower(), version.lower())
        else:
            self.modify_logger(self.trainer.logger, exp_name.lower(), version.lower())

        if self.subcommand in ["validate", "test"]:
            self.config_init[self.subcommand]["verbose"] = False

    before_fit = before_validate = before_test = before_run

    def after_run(self):
        results = {}

        if self.trainer.state.fn == TrainerFn.FITTING:
            if (
                self.trainer.checkpoint_callback
                and self.trainer.checkpoint_callback.best_model_path
            ):
                ckpt_path = self.trainer.checkpoint_callback.best_model_path
                # Disable useless logging
                logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(
                    logging.WARNING
                )
                logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(
                    logging.WARNING
                )

                self.trainer.callbacks = []
                fn_kwargs = {
                    "model": self.model,
                    "datamodule": self.datamodule,
                    "ckpt_path": ckpt_path,
                    "verbose": False,
                }
                has_val_loader = (
                    self.trainer._data_connector._val_dataloader_source.is_defined()
                )
                has_test_loader = (
                    self.trainer._data_connector._test_dataloader_source.is_defined()
                )

                val_results = (
                    self.trainer.validate(**fn_kwargs) if has_val_loader else []
                )
                test_results = self.trainer.test(**fn_kwargs) if has_test_loader else []

                results = dict(ChainMap(*val_results, *test_results))
        else:
            results = self.trainer.logged_metrics

        if results:
            results_str = json.dumps(results, ensure_ascii=False, indent=2)
            print(results_str)

            metrics_file = Path(self.trainer.log_dir) / "metrics.json"
            with metrics_file.open("w") as f:
                f.write(results_str)

    after_fit = after_validate = after_test = after_run

    def setup_parser(
        self,
        add_subcommands: bool,
        main_kwargs: dict[str, Any],
        subparser_kwargs: dict[str, Any],
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser(**main_kwargs)
        shtab.add_argument_to(self.parser, ["-s", "--print-completion"])

        if add_subcommands:
            self._subcommand_method_arguments: dict[str, list[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)
