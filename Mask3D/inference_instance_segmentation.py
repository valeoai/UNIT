import os
import torch
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from trainer.trainer import InstanceSegmentation
from pytorch_lightning import Trainer, seed_everything


def get_parameters(cfg: DictConfig):
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    assert cfg.general.checkpoint is not None, "backbone_checkpoint is None"

    model = InstanceSegmentation(cfg)
    
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    model.load_state_dict(state_dict)

    model.eval()

    return cfg, model


@hydra.main(
    config_path="conf", config_name="config_base_inference.yaml"
)
def inference(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model = get_parameters(cfg)
    runner = Trainer(
        accelerator="gpu",
        devices=cfg.general.gpus,
        precision=cfg.general.precision,
        **cfg.trainer,
    )
    runner.predict(model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(
    config_path="conf", config_name="config_base_inference.yaml"
)
def main(cfg: DictConfig):
    inference(cfg)


if __name__ == "__main__":
    main()
