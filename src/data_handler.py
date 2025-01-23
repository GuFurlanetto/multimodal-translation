from typing import List, Optional, Sequence, Union, Any
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as ta
from torch import nn
import numpy as np
import torchaudio
import torch
import os
import glob
import cv2
import yaml
import tqdm
import soundfile as sf

import librosa


class InstrumentDataset(Dataset):
    """
    Main class for loading the Instrument data

    Param (init)
        datapath -> Str: Path to the main folder containing the data splited

        mode -> Str [image2audio, audio2image]: Model mode for getting (example, label) pair

    """

    def __init__(self, datapath, mode, split):
        super().__init__()

        assert mode in ["audio2image", "image2audio"]

        self.audio_files = sorted(glob.glob(f"{datapath}/{mode}/{split}/*.wav"))
        self.image_files = sorted(glob.glob(f"{datapath}/{mode}/{split}/*.png"))

        self.mode = mode
        self.split = split

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        # Get instruments image example
        image = self.image_files[index]
        img_original = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_final = cv2.resize(img_original, (128, 128))

        # Load audio and resample to 48000 Hz
        waveform, sample_rate = librosa.load(self.audio_files[index], sr=48000)

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

        # mel_spectrogram = librosa.util.normalize(mel_spectrogram)

        if self.mode == "image2audio":
            return (
                torch.tensor(image_final, dtype=torch.float32).unsqueeze(0),
                torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0),
            )
        else:
            return torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(
                0
            ), torch.tensor(image_final, dtype=torch.float32).unsqueeze(0)


class ImageAudioPairDatset(Dataset):
    """
    Main class for loading the VGG data

    Param (init)
        datapath -> Str: Path to the main folder containing the data splited

        mode -> Str [image2audio, audio2image]: Model mode for getting (example, label) pair
    """

    def __init__(self, datapath, mode, split):
        super().__init__()

        self.audio_files = sorted(glob.glob(f"{datapath}/{split}/*.flac"))
        self.video_files = sorted(glob.glob(f"{datapath}/{split}/*.jpg"))

        assert mode in ["audio2image", "image2audio"]
        self.mode = mode
        self.split = split

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        image_file = self.video_files[index]

        # Assert we are getting the right pair of examples
        assert os.path.basename(audio_file)[:-5] == os.path.basename(image_file)[:-4]

        # Process audio
        # In our system the audo file should have
        waveform, sample_rate = torchaudio.load(audio_file)

        resampler = torchaudio.transforms.Resample(sample_rate, 480000)
        waveform = resampler(waveform)

        # To make the audio spectogram be the same size as the image
        # here we cut or pad the audio when necessary
        pad = 0
        if waveform.shape[1] > sample_rate:
            waveform = waveform[:, :480000]
        elif waveform.shape[1] < sample_rate:
            pad = int(((sample_rate - waveform.shape[1]) / 2))

        # if Stereo audio, transform it to Mono
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)

        # Mel spectogram stransformation
        mel_transformation = ta.MelSpectrogram(
            sample_rate=sample_rate, n_fft=7500, n_mels=128
        )
        mel_spec = mel_transformation(waveform)[:, :, :-1]
        if mel_spec.shape[-1] != 128:
            print(audio_file)

        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        # resize image
        resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        if self.mode == "image2audio":
            return torch.tensor(resized_img, dtype=torch.float32).unsqueeze(0), mel_spec
        elif self.mode == "audio2image":
            return mel_spec, torch.tensor(resized_img, dtype=torch.float32).unsqueeze(0)

    def process_image(self, image):
        """
        Resize the image to (128, 128) and convert it to Gray scale

        Args
            Image -> np.array: Image to be processed

        Return
            Image -> np.array: Gray scale image resized
        """
        image_resized = cv2.resize(image, (128, 128))
        imate_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        imate_gray = imate_gray / 255

        return torch.tensor(imate_gray).unsqueeze(0)

    def extract_frames(self, video_file):
        """
        Extract all frames from the video. This functions
        also apply all transformation needed for our model

        Args
            video_file -> Str: Video file path

        Return
            frames -> List[np.array()]: List of frame sin the video
        """
        video_capture = cv2.VideoCapture(video_file)
        success, image = video_capture.read()

        frames = []
        while success:
            frames.append(self.process_image(image))
            success, image = video_capture.read()

        video_capture.release()

        return frames


class MNISTMultimodal(Dataset):
    """
    Main class for loading the MNIST data

    Param (init)
        datapath -> Str: Path to the main folder containing the data splited

        mode -> Str [image2audio, audio2image]: Model mode for getting (example, label) pair

    """

    def __init__(self, datapath, mode, split):
        super().__init__()

        assert mode in ["audio2image", "image2audio"]

        self.audio_files = sorted(glob.glob(f"{datapath}/{mode}/{split}/*.wav"))
        self.image_files = sorted(glob.glob(f"{datapath}/{mode}/{split}/*.jpg"))

        self.mode = mode
        self.split = split

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        # Get MNIST image example
        image = self.image_files[index]
        img_original = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_final = cv2.resize(img_original, (128, 128))

        # Load audio and resample to 48000 Hz
        waveform, sample_rate = librosa.load(self.audio_files[index], sr=48000)

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

        if self.mode == "image2audio":
            return (
                torch.tensor(image_final, dtype=torch.float32).unsqueeze(0),
                torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0),
            )
        else:
            return torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(
                0
            ), torch.tensor(image_final, dtype=torch.float32).unsqueeze(0)


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        mode: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        data_class: object = ImageAudioPairDatset,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.mode = mode
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_class = eval(data_class)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = self.data_class(
            self.data_dir,
            self.mode,
            split="train",
        )

        self.val_dataset = self.data_class(
            self.data_dir,
            self.mode,
            split="val",
        )

        self.test_dataset = self.data_class(
            self.data_dir,
            self.mode,
            split="test",
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


import shutil


def replace_audio_file(examples_path, dataset_path):
    # Get audio examples
    audio_files = glob.glob(f"{examples_path}/*.wav")
    audio_files = {os.path.basename(file)[0]: file for file in audio_files}

    # Get ref list
    ref_audio = glob.glob(f"{examples_path}/trainingSet/*/*")
    look_up_names = [os.path.basename(file) for file in ref_audio]
    number_class = [file.split("/")[-2] for file in ref_audio]

    # replace files
    for image in glob.glob(f"{dataset_path}/*.jpg"):
        image_name = os.path.basename(image)
        image_idx = look_up_names.index(image_name)
        image_class = number_class[image_idx]

        ref_audio = audio_files[image_class]

        shutil.copyfile(
            ref_audio, f"{dataset_path}/{image_name.replace('.jpg', '.wav')}"
        )


if __name__ == "__main__":
    path = "data/instrument_data"

    data = InstrumentDataset(path, "audio2image", "val")

    max = 0
    for idx in range(data.__len__()):
        image, audio = data.__getitem__(idx)

        print(image.shape)

    print(f"[ INFO ] Max: {max}")

    # splits = ["train", "val", "test"]
    # data_path = "data/mnist_data/image2audio"
    # examples_path = "data/mnist_data/examples"

    # for split in splits:
    #     data_dir = f"{data_path}/{split}"

    #     replace_audio_file(examples_path, data_dir)
