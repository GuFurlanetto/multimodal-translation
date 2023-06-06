from pytorch_lightning.utilities.seed import seed_everything
from data_handler import VAEDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model.translation_model import BetaVAE
from pytorch_lightning.plugins import DDPPlugin
from utils import VAEXperiment
import numpy as np
import argparse
import yaml
import os
from pathlib import Path


def train_model(training_arguments):
    """
    Run a training session with the Autoencoder with the given training arguments

    Params:
       training_arguments -> Dict: Training arguments, including model paramters
    """

    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(
        save_dir=config["logging_params"]["save_dir"],
        name=config["model_params"]["name"],
    )

    # For reproducibility
    print(f"[ INFO ] Loading model...")
    seed_everything(config["exp_params"]["manual_seed"], True)

    model = BetaVAE(**config["model_params"])
    experiment = VAEXperiment(model, config["exp_params"])
    print(f"[ INFO ] Model Loaded")

    print(f"[ INFO ] Loading dataset ...")
    data = VAEDataset(
        **config["data_params"], pin_memory=len(config["trainer_params"]["gpus"]) != 0
    )
    print(f"[ INFO ] Dataset loaded")

    print(f"[ INFO ] Setting up pre training configs ...")
    data.setup()
    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            ),
        ],
        strategy=DDPPlugin(find_unused_parameters=False),
        **config["trainer_params"],
    )
    print(f"[ INFO ] Pre training configs setted")

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"[ INFO ] Training {config['model_params']['name']}")
    runner.fit(experiment, datamodule=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic runner for VAE models")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="configs/vae.yaml",
    )

    args = parser.parse_args()

    train_model(training_arguments=args)
