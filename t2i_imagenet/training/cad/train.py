import os
import shutil
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from lightning_fabric.utilities.rank_zero import _get_rank
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb
from callbacks import EMACallback, FixNANinGrad, IncreaseDataEpoch, LogGeneratedImages
from models.diffusion import DiffusionModule

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):
    # print(OmegaConf.to_yaml(cfg, resolve=True))

    if "stage" in cfg and cfg.stage == "debug":
        import lovely_tensors as lt

        lt.monkey_patch()

    dict_config = OmegaConf.to_container(cfg, resolve=True)

    Path(cfg.checkpoints.dirpath).mkdir(parents=True, exist_ok=True)

    print("Working directory : {}".format(os.getcwd()))
    shutil.copyfile(
        Path(".hydra/config.yaml"),
        f"{cfg.checkpoints.dirpath}/config.yaml",
    )

    hydra_overrides = dict([x.split("=") for x in HydraConfig.get().overrides.task])
    hydra_overrides["root_dir"] = cfg.root_dir

    log_dict = {}

    log_dict["model"] = dict_config["model"]

    log_dict["data"] = dict_config["data"]

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)

    # datamodule.setup()

    checkpoint_callback = hydra.utils.instantiate(
        cfg.checkpoints, hydra_overrides=hydra_overrides
    )

    progress_bar = hydra.utils.instantiate(cfg.progress_bar)

    ema_callback = EMACallback(
        "network",
        "ema_network",
        decay=cfg.model.ema_decay,
        start_ema_step=cfg.model.start_ema_step,
        init_ema_random=False,
    )

    log_images_callback = LogGeneratedImages(
        root_dir=cfg.root_dir,
        mode=cfg.data.type,
        num_classes=cfg.data.label_dim,
        shape=(
            cfg.model.network.num_input_channels,
            cfg.data.data_resolution,
            cfg.data.data_resolution,
        ),
        log_every_n_steps=cfg.checkpoints.every_n_train_steps,
        log_conditional=cfg.checkpoints.validate_conditional,
        log_unconditional=cfg.checkpoints.validate_unconditional,
        text_embedding_name=(
            cfg.model.text_embedding_name
            if hasattr(cfg.model, "text_embedding_name")
            else None
        ),
        batch_size=datamodule.batch_size,
        cfg_rate=cfg.model.cfg_rate if hasattr(cfg.model, "cfg_rate") else 0,
        negative_prompts=(
            cfg.model.negative_prompts
            if hasattr(cfg.model, "negative_prompts")
            else None
        ),
    )

    lr_monitor = LearningRateMonitor()

    fix_nan_callback = FixNANinGrad(
        monitor=["train/loss"],
    )

    increase_data_epoch = IncreaseDataEpoch()
    callbacks = [
        checkpoint_callback,
        progress_bar,
        ema_callback,
        log_images_callback,
        lr_monitor,
        fix_nan_callback,
        increase_data_epoch,
    ]

    # rank = _get_rank()

    # if os.path.isfile(Path(cfg.checkpoints.dirpath) / Path("wandb_id.txt")):
    #     with open(
    #         Path(cfg.checkpoints.dirpath) / Path("wandb_id.txt"), "r"
    #     ) as wandb_id_file:
    #         wandb_id = wandb_id_file.readline()
    # else:
    #     wandb_id = wandb.util.generate_id()
    #     print(f"generated id{wandb_id}")
    #     if rank == 0 or rank is None:
    #         with open(
    #             Path(cfg.checkpoints.dirpath) / Path("wandb_id.txt"), "w"
    #         ) as wandb_id_file:
    #             wandb_id_file.write(str(wandb_id))

    logger = hydra.utils.instantiate(cfg.logger)  # , id=wandb_id, r esume="allow")
    logger._wandb_init.update({"config": log_dict})
    # logger.log_hyperparams(dict_config)
    model = DiffusionModule(cfg.model)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    ckpt_path = None

    if (Path(cfg.checkpoints.dirpath) / Path("last.ckpt")).exists():
        ckpt_path = Path(cfg.checkpoints.dirpath) / Path("last.ckpt")
    elif (Path(cfg.checkpoints.dirpath) / Path("init.ckpt")).exists():
        ckpt_path = None
        model.load_state_dict(
            torch.load(
                Path(cfg.checkpoints.dirpath) / Path("init.ckpt"), map_location="cpu"
            )
        )
        print("loaded model from init ckpt")
    else:
        ckpt_path = None

    logger.experiment.watch(model, log="all", log_graph=True, log_freq=1000)

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
