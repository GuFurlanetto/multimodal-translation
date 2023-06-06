from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchaudio import transforms as ta
from utils import plot_waveform, plot_spectrogram
from utils_io import make_video
from torch import nn
import numpy as np
import torchaudio
import torch
import os
import glob
import cv2
import yaml


class ImageAudioPairDatset(Dataset):
    """
    Main class for loading the VGG data

    Param (init)
        datapath -> Str: Path to the main folder containing the data splited

        mode -> Str [image2audio, audio2image]: Model mode for getting (example, label) pair
    """

    def __init__(self, datapath, mode, split):
        super().__init__()

        self.audio_files = sorted(glob.glob(f"{datapath}/{split}*.flac"))
        self.video_files = sorted(glob.glob(f"{datapath}/{split}*.mp4"))

        assert mode in ["audio2image", "image2audio"]
        self.mode = mode
        self.split = split

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        video_file = self.video_files[index]

        # Assert we are getting the right pair of examples
        assert os.path.basename(audio_file)[:-5] == os.path.basename(video_file)[:-4]

        # Process audio
        # In our system the audo file should have
        waveform, sample_rate = torchaudio.load(audio_file)
        print(f"[ INFO ] Sample rate: {sample_rate}")

        # if Stereo audio, transform it to Mono
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)

        # Mel spectogram stransformation
        mel_transformation = ta.MelSpectrogram(sample_rate=sample_rate, n_fft=4500)
        mel_spec = mel_transformation(waveform)[:, :, :-1]

        # The image shoud be gray scale and have a fixed size of 256x256
        video_frames = self.extract_frames(video_file)
        ref_frame = video_frames[int(len(video_frames) / 2)]

        if self.mode == "image2audio":
            return ref_frame, mel_spec
        elif self.mode == "audio2image":
            return mel_spec, ref_frame

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

        return frames


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

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ImageAudioPairDatset(
            self.data_dir,
            self.mode,
            split="train",
        )

        self.val_dataset = ImageAudioPairDatset(
            self.data_dir,
            self.mode,
            split="val",
        )

        self.test_dataset = ImageAudioPairDatset(
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
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    with open("src/model/config.yml", "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    data = VAEDataset(
        **config["data_params"], pin_memory=len(config["trainer_params"]["gpus"]) != 0
    )

    import pdb

    pdb.set_trace()
