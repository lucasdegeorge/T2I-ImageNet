import os
from pathlib import Path
from typing import Any, Dict

from lightning_fabric.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint

from metrics.sample_and_eval import SampleAndEval


class ModelCheckpointValidate(ModelCheckpoint):
    """
    Run model checkpoint and run validation on each new checkpoint
    """

    def __init__(
        self,
        hydra_overrides,
        qos="t3",
        gpu_type="a100",
        validate_when_not_on_cluster=False,
        validate_when_on_cluster=False,
        eval_set="val",
        dataset_name="CIFAR-10",
        validate_conditional=True,
        validate_unconditional=False,
        validate_per_class_metrics=True,
        num_classes=10,
        shape=(4, 32, 32),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_cluster = os.getenv("IS_CLUSTER", False)
        self.hydra_overrides = hydra_overrides
        self.ready_to_validate = False
        self.qos = qos
        self.gpu_type = gpu_type
        self.root_dir = hydra_overrides.root_dir
        self.validate_when_not_on_cluster = validate_when_not_on_cluster
        self.validate_when_on_cluster = validate_when_on_cluster
        self.eval_set = eval_set
        self.dataset_name = dataset_name
        self.validate_conditional = validate_conditional
        self.validate_unconditional = validate_unconditional
        self.validate_per_class_metrics = validate_per_class_metrics
        self.num_classes = num_classes
        self.shape = shape

        if "computer.devices" in self.hydra_overrides:
            del self.hydra_overrides["computer.devices"]
        if "computer.num_nodes" in self.hydra_overrides:
            del self.hydra_overrides["computer.num_nodes"]

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)
        self.ready_to_validate = False

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)
        self.ready_to_validate = True

    def _save_checkpoint(self, trainer: "L.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if self.ready_to_validate and self.is_cluster and self.validate_when_on_cluster:
            checkpoint_name = os.path.basename(filepath)
            if checkpoint_name.startswith("epoch"):
                print(f"Validating {checkpoint_name}")
                self.validate_on_cluster(checkpoint_name, file_name="validate.py")

    def on_validation_end(self, trainer, pl_module) -> None:
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
            and self.ready_to_validate
            and not self.is_cluster
            and self.validate_when_not_on_cluster
        ):
            metric = SampleAndEval(
                pl_module.logger,
                root_dir=self.root_dir,
                compute_conditional=self.validate_conditional,
                compute_unconditional=self.validate_unconditional,
                compute_per_class_metrics=self.validate_per_class_metrics,
                log_prefix="val",
                eval_set=self.eval_set,
                dataset_name=self.dataset_name,
                num_classes=self.num_classes,
                shape=self.shape,
                cfg_rate=(
                    pl_module.cfg.cfg_rate if hasattr(pl_module.cfg, "cfg_rate") else 0
                ),
            )
            metric.compute_and_log_metrics(
                pl_module.device,
                pl_module,
                trainer.datamodule,
                step=pl_module.global_step,
                epoch=pl_module.current_epoch,
            )
        super().on_validation_end(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._every_n_train_steps >= 1
            and (trainer.global_step % self._every_n_train_steps == 0)
            and not self.is_cluster
            and self.validate_when_not_on_cluster
        ):
            metric = SampleAndEval(
                pl_module.logger,
                root_dir=self.root_dir,
                compute_conditional=self.validate_conditional,
                compute_unconditional=self.validate_unconditional,
                compute_per_class_metrics=self.validate_per_class_metrics,
                log_prefix="val",
                eval_set=self.eval_set,
                dataset_name=self.dataset_name,
                num_classes=self.num_classes,
                shape=self.shape,
                cfg_rate=(
                    pl_module.cfg.cfg_rate if hasattr(pl_module.cfg, "cfg_rate") else 0
                ),
            )
            metric.compute_and_log_metrics(
                pl_module.device,
                pl_module,
                trainer.datamodule,
                step=pl_module.global_step,
                epoch=pl_module.current_epoch,
            )
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_test_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        metric = SampleAndEval(
            pl_module.logger,
            root_dir=self.root_dir,
            compute_conditional=self.validate_conditional,
            compute_unconditional=self.validate_unconditional,
            compute_per_class_metrics=self.validate_per_class_metrics,
            log_prefix="test",
            eval_set=self.eval_set,
            dataset_name=self.dataset_name,
            num_classes=self.num_classes,
            shape=self.shape,
            cfg_rate=(
                pl_module.cfg.cfg_rate if hasattr(pl_module.cfg, "cfg_rate") else 0
            ),
        )
        metric.compute_and_log_metrics(
            pl_module.device,
            pl_module,
            trainer.datamodule,
            step=pl_module.global_step,
            epoch=pl_module.current_epoch,
        )
