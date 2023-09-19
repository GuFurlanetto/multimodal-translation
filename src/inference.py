import torch
import numpy as np
from model.model_zoo.beta_vae import BetaVAE
from model.model_zoo.wae_mmd import WAE_MMD
import cv2
import torchaudio
import argparse
import yaml
import glob
import os
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_data(data, mode):
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
            image_final = cv2.resize(img_original, (24, 24)) / 255

            yield torch.tensor(image_final, dtype=torch.float32).to(device).unsqueeze(
                0
            ).unsqueeze(0)
    else:
        for file in sorted(data):
            waveform, sample_rate = torchaudio.load(file)

            # if Stereo audio, transform it to Mono
            if waveform.shape[0] == 2:
                waveform = torch.mean(waveform, dim=0).unsqueeze(0)

            # Standard sample at 8000
            resampler = torchaudio.transforms.Resample(sample_rate, 8000)
            waveform = resampler(waveform)

            # To make the audio spectogram be the same size as the image
            # here we cut or pad the audio when necessary
            pad = 0
            if waveform.shape[1] >= 8000:
                waveform = waveform[:, :8000]
            elif waveform.shape[1] < sample_rate:
                pad = int(((8000 - waveform.shape[1]) / 2))

            # Extract the mel Spectogram with 24 mel banks, it generates a  24x24 spec
            mel_transfomer = torchaudio.transforms.MelSpectrogram(
                sample_rate, n_mels=24, n_fft=664, pad=pad
            )
            mel_spec = mel_transfomer(waveform)[:, :, :-1]

            yield torch.tensor(mel_spec).to(device).unsqueeze(0)


def run_inference(args):
    """
    Run inference on a given dataset

    Args:
        args -> *args: Arguments from command line -> {dataset path, weights path, config path}
    """

    # Load data
    print("[ INFO ] Loading data ...")
    input_data = sorted(glob.glob(f"{args.inference_data}/*.jpg"))

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # Process data depending on format (image2audio or audio2image)
    data = process_data(input_data, config["data_params"]["mode"])
    print("[ INFO ] Data loaded")

    # Load model
    print("[ INFO ] Loading model ...")
    model = WAE_MMD(**config["model_params"]).to(device)
    checkpoint = torch.load(args.weights)

    # Remove prefix "model." from layers name
    checkpoint["state_dict"] = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("[ INFO ] Model loaded")

    output_dir = "./output"
    if args.output_dir:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("[ INFO ] Running inference ...")
    for idx, file in tqdm.tqdm(enumerate(data)):
        result = model(file)[0].cpu().detach().numpy().squeeze()

        # Unormalize result
        result = result * 255

        cv2.imwrite(f"{output_dir}/{idx + 1}.jpg", result)
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
