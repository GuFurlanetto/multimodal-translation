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
            image_final = cv2.resize(img_original, (image_size, image_size)) / 255

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
    if config["data_params"]["mode"] == "image2audio":
        input_data = sorted(glob.glob(f"{args.inference_data}/*.jpg"))
    else:
        input_data = sorted(glob.glob(f"{args.inference_data}/*.wav"))

    # Process data depending on format (image2audio or audio2image)
    data = process_data(
        input_data,
        config["data_params"]["mode"],
        config["data_params"]["image_size"],
        config["data_params"]["sample_rate"],
        config["data_params"]["n_mels"],
        config["data_params"]["n_fft"],
    )
    print("[ INFO ] Data loaded")

    # Load model
    print("[ INFO ] Loading model ...")
    model_audio_2_image = WAE_MMD(**config["model_params"])
    checkpoint = torch.load("best_models/mnist/audio2image/checkpoints/last.ckpt")
    checkpoint["state_dict"] = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
    model_audio_2_image.load_state_dict(checkpoint["state_dict"])

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_image_2_audio = WAE_MMD(**config["model_params"])
    checkpoint = torch.load("best_models/mnist/image2audio/checkpoints/last.ckpt")
    checkpoint["state_dict"] = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
    model_image_2_audio.load_state_dict(checkpoint["state_dict"])

    # Remove prefix "model." from layers name
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model = WAE_MMD(**config["model_params"]).to(device)

    model.encoder.load_state_dict(model_image_2_audio.encoder.state_dict())
    model.fc_mu.load_state_dict(model_image_2_audio.fc_mu.state_dict())
    model.fc_var.load_state_dict(model_image_2_audio.fc_var.state_dict())

    model.decoder_input.load_state_dict(model_audio_2_image.decoder_input.state_dict())
    model.decoder.load_state_dict(model_audio_2_image.decoder.state_dict())
    model.final_layer.load_state_dict(model_audio_2_image.final_layer.state_dict())
    model.eval()

    print("[ INFO ] Model loaded")

    output_dir = "./output"
    if args.output_dir:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("[ INFO ] Running inference ...")
    for idx, (file, file_path) in tqdm.tqdm(enumerate(zip(data, input_data))):
        result = model(file)[0].cpu().detach().numpy().squeeze()

        # Unormalize result

        file_name = os.path.basename(file_path)[:-4]

        if config["data_params"]["mode"] == "audio2image":
            # v2.imwrite(f"{output_dir}/{file_name}.png", result)
            # Transform spectogram back to waveform
            process_audio_results(
                result,
                f"{output_dir}/{file_name}.wav",
                config["data_params"]["sample_rate"],
                config["data_params"]["n_fft"],
                config["data_params"]["hop_lenght"],
            )
        else:
            result = result * 255
            cv2.imwrite(f"{output_dir}/{file_name}.png", result)
    print("[ INFO ] Inference complete")


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
