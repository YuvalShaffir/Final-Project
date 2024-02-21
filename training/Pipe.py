#requirments
# !pip install diffusers
# !pip install torch
# !pip install accelerate
import inspect
from typing import List

import scipy
import os
from diffusers import AudioLDM2Pipeline
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from transformers import SpeechT5HifiGan
import torch
from IPython.display import Audio
from accelerate import Accelerator

ETA_VALUE = 0.0

GENERATOR = "generator"

ETA = "eta"


class Pipe:
    """
    This class is a pipeline of the model
    """
    def __init__(self, path_to_models_dir: str):
        self.path_to_models_dir = path_to_models_dir
        # path_to_model = "/content/drive/My Drive/final_unet"
        # from diffusers.models import AutoencoderKL

        scheduler_path, unet_path, vae_path, vocoder_path = self._get_model_paths()

        self.p_device = self._choose_device()

        self.vae, self.vocoder, self.scheduler, self.unet = self._load_models(scheduler_path, unet_path, vae_path,
                                                                              vocoder_path)

        accelerator = Accelerator()

        self.vae, self.vocoder, self.scheduler, self.unet = accelerator.prepare(self.vae, self.vocoder,
                                                                                self.scheduler, self.unet)

    def _get_model_paths(self):
        """ Gets the paths to each model """
        try:
            [unet_path, scheduler_path, vae_path, vocoder_path] = self._get_dir_paths(directory="models")
            # print(f"models paths: \n {unet_path} \n {scheduler_path} \n {vae_path} \n {vocoder_path} \n")
            return [scheduler_path, unet_path, vae_path, vocoder_path]
        except Exception as e:
            print(f"Error in _get_model_paths(), description: {e}")

    def _load_models(self, scheduler_path: str, unet_path: str, vae_path: str, vocoder_path: str):
        """ Load the models """
        try:
            vae = torch.load(vae_path, map_location=self.p_device, weights_only=False).eval()
            vocoder = torch.load(vocoder_path, map_location=self.p_device).eval()
            scheduler = torch.load(scheduler_path)
            unet = torch.load(unet_path).to(self.p_device)
            # print(f"models: \n {vae} \n {vocoder} \n {scheduler} \n {unet} \n")
            return vae, vocoder, scheduler, unet
        except Exception as e:
            print(f"Error in _load_models(), description: {e}")

    def _choose_device(self):
        if torch.cuda.is_available():
            p_device = 'cuda'
        else:
            p_device = 'cpu'
        return p_device

    def _prepare_latents(self, num_waveform):
        return

    def _encode_latents(self, latents):
        """
        Given a tensor of type float16, return a tensor of type float16
        :param latents_to_encode:
        :return:
        """
        return self.vae.encode(latents).latent_dist.sample() * 0.18215

    def _mel_spectrogram_to_waveform(self, mel_spectrogram):
        """
        Convert a mel spectrogram to waveform using the vocoder.

        Args:
            mel_spectrogram (torch.Tensor): Mel spectrogram tensor.

        Returns:
            torch.Tensor: Waveform tensor.
        """
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
            print("was here")
        # check
        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    def _prepare_extra_step_kwargs(self, generator, eta):
        """
        prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        and should be between [0, 1]

        @param generator: the generator model
        @param eta: the value of η
        @return: a dictionary of extra kwargs
        """

        accepts_eta = ETA in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs[ETA] = eta

        # check if the scheduler accepts generator
        accepts_generator = GENERATOR in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs[GENERATOR] = generator
        return extra_step_kwargs

    def _inference(self, extra_step_kwargs, latents, n):
        """
        Perform inference using the trained model.

        Args:
            extra_step_kwargs (dict): Extra keyword arguments for the inference step.
            latents (torch.Tensor): Latent vectors.
            n (torch.Tensor): Encoder hidden states.

        Returns:
            torch.Tensor: Updated latent vectors after inference.
        """
        num_inference_steps = 200
        self.scheduler.set_timesteps(num_inference_steps, 'cuda')
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        for i, t in enumerate(timesteps):
            print(i)
            # expand the latents if we are doing classifier free guidance
            # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latents, timestep=t, encoder_hidden_states=n, return_dict=False)[0]
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    def _get_dir_paths(self, directory: str) -> List[str]:
        """
        :return: Returns to paths to all the files in the given directory
        """
        file_paths = []
        directory_full_path = os.path.abspath(directory)
        # Walk through all files and directories in the given directory
        for root, directories, files in os.walk(directory_full_path):
            for filename in files:
                # Construct the full path to the file
                file_path = os.path.join(root, filename)
                # Append the file path to the list
                file_paths.append(file_path)
        return file_paths

    def run_pipe(self):
        eta = ETA_VALUE
        generator = torch.Generator(device='cuda')
        extra_step_kwargs = self._prepare_extra_step_kwargs(generator, eta)
        tensor = torch.randn((1, 1, 1024, 64), dtype=torch.float16, device=self.p_device)
        latents = self._encode_latents(tensor)
        n = torch.randn((1, 64, 1024), dtype=torch.float32, device=self.p_device)
        self._inference(extra_step_kwargs, latents, n)
        log_mels = self.vae.decode(latents.type(torch.float16))

        original_waveform_length = 163872
        audio = self._mel_spectrogram_to_waveform(log_mels.sample)
        audio = audio[:, :original_waveform_length].detach().numpy()
        # display audio
        Audio(audio, rate=16000)

