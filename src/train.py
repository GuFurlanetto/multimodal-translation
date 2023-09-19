from pytorch_lightning.utilities.seed import seed_everything
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from data_handler import VAEDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from model.model_zoo.beta_vae import BetaVAE
from model.model_zoo.wae_mmd import WAE_MMD
from pytorch_lightning.plugins import DDPPlugin
from utils import VAEXperiment
import numpy as np
import argparse
import yaml
import os
from pathlib import Path
from utils import load_model
from utils_io import load_yaml


def train_model(training_arguments):
    """
    Run a training session with the Autoencoder with the given training arguments

    Params:
       training_arguments -> Dict: Training arguments, including model paramters
    """

    # Load config file
    config = load_yaml(args.config_file)
    log_dir = args.log_dir

    # Init logger
    mlflow_logger = MLFlowLogger(experiment_name=args.exp_name, run_name=args.run_name)

    # Loading model
    print(f"[ INFO ] Loading model...")
    seed_everything(config["exp_params"]["manual_seed"], True)

    model = load_model(config, args)
    experiment = VAEXperiment(model, config["exp_params"], log_dir)
    print(f"[ INFO ] Model Loaded")

    # Loading dataset
    print(f"[ INFO ] Loading dataset ...")
    data = VAEDataset(
        **config["data_params"], pin_memory=len(config["trainer_params"]["gpus"]) != 0
    )
    print(f"[ INFO ] Dataset loaded")

    print(f"[ INFO ] Setting up pre training configs ...")
    data.setup()
    runner = Trainer(
        logger=mlflow_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(args.log_dir, "checkpoints"),
                monitor="Reconstruction_Loss",
                save_last=True,
            ),
        ],
        strategy=DDPPlugin(find_unused_parameters=False),
        **config["trainer_params"],
    )
    print(f"[ INFO ] Pre training configs setted")

    Path(f"{log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    # Running training
    print(f"[ INFO ] Training {config['model_params']['name']}")
    runner.fit(experiment, datamodule=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic runner for VAE models")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        metavar="FILE",
        help="path to the config file",
        required=True,
    )

    parser.add_argument(
        "-l",
        "--log-dir",
        dest="log_dir",
        help="Path where logs will be saved",
        required=True,
    )

    parser.add_argument(
        "-m",
        "--model",
        dest="model_name",
        help="Model name",
        required=True,
        choices=["beta_vae", "wae_mmd"],
    )

    parser.add_argument(
        "-e",
        "--exp-name",
        dest="exp_name",
        help="Mlflow experiment name for current run",
        default="Training model",
    )

    parser.add_argument(
        "-r",
        "--run-name",
        dest="run_name",
        help="Run name for current training session",
        default="Default run",
    )

    args = parser.parse_args()

    train_model(training_arguments=args)
