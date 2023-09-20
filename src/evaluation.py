import argparse
import pytorch_lightning
import model.model_zoo
from utils_io import load_yaml, save_eval
from utils import load_model
from data_handler import VAEDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import os
import tqdm
import torch
from datetime import datetime
import time
import tqdm
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(args):
    print(f"[ INFO ] Setting up pre training configs ...")
    config_file = load_yaml(args.config_file)

    # Loading dataset
    print(f"[ INFO ] Loading dataset ...")
    data = VAEDataset(
        **config_file["data_params"],
        pin_memory=len(config_file["trainer_params"]["gpus"]) != 0,
    )
    data.setup()
    test_data = data.test_dataloader()
    print(f"[ INFO ] Dataset loaded")

    # Loading model
    print("[ INFO ] Loading model ...")
    model = load_model(config_file, args).to(device)
    checkpoint = torch.load(args.weights_path)

    # Remove prefix "model." from layers name
    checkpoint["state_dict"] = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("[ INFO ] Model loaded")

    # Performs evaluation on test data
    print("[ INFO ] Running evaluation ...")
    metrics = defaultdict(lambda: [])
    for idx, (input, target) in tqdm.tqdm(enumerate(test_data)):
        input, target = (
            input.to(device),
            target.to(device).cpu().detach().numpy().squeeze(),
        )

        predicts = model(input)[0].cpu().detach().numpy().squeeze()

        # Calculates metrics
        metrics["ssim_value"].extend(
            [
                ssim(img1, img2, data_range=img1.max() - img1.min())
                for img1, img2 in zip(predicts, target)
            ]
        )

        metrics["mse"].extend([mse(img1, img2) for img1, img2 in zip(predicts, target)])

    print("[ INFO ] Evaluation finished")

    # Print and save results
    s = datetime.now()
    current_time = s.strftime("%d-%m-%Y_%H-%M-%S")
    folder_name = f"{args.model_name}-{current_time}"
    os.makedirs("eval_results", exist_ok=True)
    os.makedirs(f"eval_results/{folder_name}", exist_ok=True)

    print("[ INFO ] Metrics:")
    with open(f"eval_results/{folder_name}/metrics.txt", "w+") as f:
        for metric, value in metrics.items():
            mean_metric = sum(value) / len(value)
            print(f"[ INFO ] {metric}: {mean_metric}")
            f.write(f"{metric}: {mean_metric}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w",
        "--weights",
        dest="weights_path",
        help="Path to the model weights to be used during evaluation",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        help="Path to config file of the model",
        required=True,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        help="Show logs on terminal",
        default=0,
        type=int,
    )

    parser.add_argument(
        "-m",
        "--model",
        dest="model_name",
        help="Model name",
        required=True,
        choices=["beta_vae", "wae_mmd"],
    )

    args = parser.parse_args()

    eval(args)
