import pytorch_lightning as pl
import matplotlib.pyplot as plt
from model.model_zoo.base import BaseVAE
from model.model_zoo import WAE_MMD, BetaVAE
import torchvision.utils as vutils
from torch import optim
import os
from model.types_vae import *
import librosa
import torch
from train_utils.metrics import SSIM, MSE
import soundfile as sf


class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict, log_dir) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.log_dir = log_dir
        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except:
            pass

        # metrics
        self.ssim = SSIM()
        self.mse = MSE()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params["kld_weight"],  # al_img.shape[0]/ self.num_train_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
            target=labels,
        )

        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        self.ssim(results, labels)
        self.mse(results, labels)

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
            target=labels,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        )

    def on_validation_end(self) -> None:
        self.sample_images()

    def on_train_epoch_end(self) -> None:
        self.log("train_ssim", self.ssim)
        self.log("train_mse", self.mse)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(
            recons.data,
            os.path.join(
                self.log_dir,
                "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
            normalize=True,
            nrow=12,
        )

        try:
            samples = self.model.sample(144, self.curr_device, labels=test_label)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.log_dir,
                    "Samples",
                    f"{self.logger.name}_Epoch_{self.current_epoch}.png",
                ),
                normalize=True,
                nrow=12,
            )
        except Warning:
            pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params["LR_2"] is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model, self.params["submodel"]).parameters(),
                    lr=self.params["LR_2"],
                )
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params["scheduler_gamma"] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0], gamma=self.params["scheduler_gamma"]
                )
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params["scheduler_gamma_2"] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1], gamma=self.params["scheduler_gamma_2"]
                        )
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


def plot_waveform(waveform, sample_rate):
    """
    Plot waveform

    Args:
        waveform -> np.array: Auido waveform to plot

        Sample_rate -> Int: Sample rate of the audio waveform

    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show()


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    """
    Plot the specgram

    Args:
        specgram -> np.array: Spectogram to plot

        titile -> Str: Window title

        ylabel -> Str: Label of the Y axis
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=True)


def load_model(config, args):
    """
    Load model from a different set of models

    Args:
        config -> Dict: Config file loaded

        args -> argparse object: Arguments for training given from the command line

    Return:
        model -> model_class: Return model loaded with given configs
    """

    if args.model_name == "wae_mmd":
        model = WAE_MMD(**config["model_params"])
    elif args.model_name == "beta_wae":
        model = BetaVAE(**config["model_params"])
    else:
        print(f"Model {args.model_name} doesn't exist")
        exit()

    return model


def process_audio_results(generated_audio, output_file, sample_rate, n_fft, hop_lenght):
    """
    Takes the generated spectograms and produces a audio file at the output folder

    Args:
        generated_audios -> np.array: Spectogram

        output_folder -> Str: Path to output folder

    Return:
        Status -> bool: True if process was completed with no errors
    """

    try:
        # Invert the Mel spectrogram to the linear scale
        audio_signal = librosa.feature.inverse.mel_to_audio(
            generated_audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_lenght,
            norm="slaney",
        )

        # Save the audio to a file
        sf.write(output_file, audio_signal, sample_rate, "PCM_24")

        return True
    except:
        return False
