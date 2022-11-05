import os
from typing import Iterable

from pytorch_lightning.cli import LightningCLI

from src.callbacks.evaluator import empty_dataloader, empty_fun


class LitCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand]

        # HACK: https://github.com/Lightning-AI/lightning/issues/15233
        if config.trainer.fast_dev_run:
            config.trainer.logger = None

        logger = config.trainer.logger
        if logger and logger != True:
            loggers = logger if isinstance(logger, Iterable) else [logger]
            for logger in loggers:
                logger.init_args.save_dir = os.path.join(
                    logger.init_args.get("save_dir", "results"), self.subcommand
                )
                exp_name = config.model.class_path.split(".")[-1].lower()
                if hasattr(config, "data"):
                    data_name = config.data.class_path.split(".")[-1].lower()
                    if data_name == "blocking":
                        data_name = config.data.init_args.data_dir.split(os.sep)[
                            -1
                        ].lower()
                    exp_name = f"{exp_name}/{data_name}"
                if hasattr(logger.init_args, "name"):
                    logger.init_args.name = exp_name

    def before_run(self):
        self.model.validation_step = self.model.test_step = empty_fun
        self.datamodule.val_dataloader = empty_dataloader
        self.datamodule.test_dataloader = empty_dataloader

    before_fit = before_validate = before_test = before_run


def lit_cli():
    LitCLI(
        parser_kwargs={
            cmd: {
                "default_config_files": ["configs/presets/default.yaml"],
            }
            for cmd in ["fit", "validate", "test"]
        },
        save_config_kwargs={"overwrite": True},
    )


def get_cli_parser():
    # provide cli.parser for shtab.
    #
    # shtab --shell {bash,zsh,tcsh} src.utils.lit_cli.get_cli_parser
    # for more details see https://docs.iterative.ai/shtab/use/#cli-usage
    from jsonargparse import capture_parser

    parser = capture_parser(lit_cli)
    return parser


if __name__ == "__main__":
    lit_cli()
