#requirments
# !pip install diffusers
# !pip install torch
# !pip install accelerate
import inspect

import scipy
import os
# from diffusers import AudioLDM2Pipeline
# from diffusers.models import AutoencoderKL
# from diffusers import StableDiffusionPipeline
# from transformers import SpeechT5HifiGan
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
    def __int__(self, path_to_models_dir):
        self.path_to_models_dir = path_to_models_dir
        # path_to_model = "/content/drive/My Drive/final_unet"
        # from diffusers.models import AutoencoderKL
        vae_path = os.path.join(self.path_to_models_dir, 'vae')
        unet_path = os.path.join(self.path_to_models_dir, 'final_unet')
        vocoder_path = os.path.join(self.path_to_models_dir, 'vocoder')
        scheduler_path = os.path.join(self.path_to_models_dir, 'scheduler')
        self.p_device = self._choose_device()
        self.vae = torch.load(vae_path, map_location=p_device, weights_only=False).eval()
        self.vocoder = torch.load(vocoder_path, map_location=p_device).eval()
        self.scheduler = torch.load(scheduler_path)
        self.unet = torch.load(unet_path).evalwww().to(p_device)
        accelerator = Accelerator()
        self.vae, self.vocoder, self.scheduler, self.unet = accelerator.prepare(self.vae, self.vocoder, self.scheduler,
                                                                                self.unet)

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
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
            print("was here")

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
