import torch
import numpy as np
from model.model_zoo.beta_vae import BetaVAE
from model.model_zoo.wae_mmd import WAE_MMD
from utils import process_audio_results
import cv2
import torchaudio
import argparse
import yaml
import glob
import os
import tqdm
import torch
from torch.nn import Flatten
import torch.nn as nn
import librosa
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_data(data, mode, image_size, sample_rate_data, n_mels, n_fft):
    """
    Process data to the format accepted by the model

    Args:
        data -> list[str]: List containing path to the examples

        mode -> str: Translation mode {audio2image, image2audio}

    Return:
        example -> generator: Generator for the data processed
    """
    if mode == "image2audio":
        for file in sorted(data):
            img_original = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            image_final = cv2.resize(img_original, (image_size, image_size))

            yield torch.tensor(image_final, dtype=torch.float32).to(device).unsqueeze(
                0
            ).unsqueeze(0)
    else:
        for file in sorted(data):
            waveform, sample_rate = librosa.load(file, sr=48000)

            # waveform = librosa.util.normalize(waveform)

            # Ensure waveform is 48000 samples long
            if len(waveform) < 48000:
                pad_width = 48000 - len(waveform)
                waveform = np.pad(waveform, (0, pad_width))
            elif len(waveform) > 48000:
                waveform = waveform[:48000]

            # Compute the Mel spectrogram with 128 mel bands, using a window of 2048 samples
            mel_spectrogram = librosa.feature.melspectrogram(
                y=waveform,
                sr=sample_rate,
                n_mels=128,
                n_fft=1024,
                hop_length=377,
                norm="slaney",
            )

            yield torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(
                0
            ).unsqueeze(0).to(device)


def kl_divergence(mu1, logvar1, mu2, logvar2):
    kl = 0.5 * torch.sum(
        logvar2
        - logvar1
        + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2)
        - 1,
        dim=1,
    )
    return kl.mean().item()


def run_inference(args):
    """
    Run inference on a given dataset

    Args:
        args -> *args: Arguments from command line -> {dataset path, weights path, config path}
    """
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    print("[ INFO ] Loading data ...")
    input_data_image = sorted(glob.glob(f"{args.inference_data}/*.jpg"))
    input_data_audio = sorted(glob.glob(f"{args.inference_data}/*.wav"))

    # Process data depending on format (image2audio or audio2image)
    data_image = process_data(
        input_data_image,
        "image2audio",
        config["data_params"]["image_size"],
        config["data_params"]["sample_rate"],
        config["data_params"]["n_mels"],
        config["data_params"]["n_fft"],
    )

    data_audio = process_data(
        input_data_audio,
        "audio2image",
        config["data_params"]["image_size"],
        config["data_params"]["sample_rate"],
        config["data_params"]["n_mels"],
        config["data_params"]["n_fft"],
    )
    print("[ INFO ] Data loaded")

    # Load model

    model_audio_2_image = WAE_MMD(**config["model_params"]).to(device).eval()
    checkpoint = torch.load("best_models/mnist/audio2image/checkpoints/last.ckpt")
    checkpoint["state_dict"] = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
    model_audio_2_image.load_state_dict(checkpoint["state_dict"])

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_image_2_audio = WAE_MMD(**config["model_params"]).to(device)
    checkpoint = torch.load("best_models/mnist/image2audio/checkpoints/last.ckpt")
    checkpoint["state_dict"] = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
    model_image_2_audio.load_state_dict(checkpoint["state_dict"])
    print("[ INFO ] Model loaded")

    all_z1_mean, all_z1_logvar = [], []
    all_z2_mean, all_z2_logvar = [], []

    for audio_input, image_input in zip(data_audio, data_image):
        z1_mean, z1_logvar = model_audio_2_image.encode(audio_input)
        z2_mean, z2_logvar = model_image_2_audio.encode(image_input)

        all_z1_mean.append(z1_mean)
        all_z1_logvar.append(z1_logvar)
        all_z2_mean.append(z2_mean)
        all_z2_logvar.append(z2_logvar)

    # Concatenate all batches
    all_z1_mean = torch.cat(all_z1_mean, dim=0)
    all_z1_logvar = torch.cat(all_z1_logvar, dim=0)
    all_z2_mean = torch.cat(all_z2_mean, dim=0)
    all_z2_logvar = torch.cat(all_z2_logvar, dim=0)

    # Compute statistical measures (e.g., KL divergence, JS divergence, EMD)
    kl_div = kl_divergence(all_z1_mean, all_z1_logvar, all_z2_mean, all_z2_logvar)
    emd = wasserstein_distance(
        all_z1_mean.flatten().cpu().detach().numpy(),
        all_z2_mean.flatten().cpu().detach().numpy(),
    )

    print(f"KL Divergence: {kl_div}")
    print(f"Earth Mover's Distance (EMD): {emd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        dest="inference_data",
        help="Path to inference data",
        required=True,
    )

    parser.add_argument(
        "-w", "--weights", dest="weights", help="Path to model wights", required=True
    )

    parser.add_argument(
        "-c", "--config", dest="config", help="Path to model configs", required=True
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        help="Path to output folder",
        required=False,
    )

    args = parser.parse_args()

    run_inference(args)
