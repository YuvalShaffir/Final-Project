#requirments
# !pip install diffusers
# !pip install torch
# !pip install accelerate
import scipy
import os
# from diffusers import AudioLDM2Pipeline
# from diffusers.models import AutoencoderKL
# from diffusers import StableDiffusionPipeline
# from transformers import SpeechT5HifiGan
import torch
from accelerate import Accelerator

class Pipe():
    def __int__(self,path_to_models_dir):
        self.path_to_models_dir = path_to_models_dir



    def load_models(self,):
        # path_to_model = "/content/drive/My Drive/final_unet"
        # from diffusers.models import AutoencoderKL
        if torch.cuda.is_available():
            p_device = ('cuda')
        else:
            p_device = ('cpu')
        vae_path= os.path.join(self.path_to_models_dir,'vae')
        unet_path = os.path.join(self.path_to_models_dir, 'final_unet')
        vocoder_path = os.path.join(self.path_to_models_dir, 'vocoder')
        scheduler_path = os.path.join(self.path_to_models_dir, 'scheduler')
        self.vae = torch.load(vae_path, map_location=p_device, weights_only=False).eval()
        self.vocoder = torch.load(vocoder_path, map_location=p_device).eval()
        self.scheduler = torch.load(scheduler_path)
        self.unet = torch.load(unet_path).eval().to(p_device)
        accelerator = Accelerator()
        self.vae, self.vocoder, self.scheduler, self.unet = accelerator.prepare(self.vae, self.vocoder, self.scheduler, self.unet)



    def prepare_latents(self,num_waveform):
        return

    #gets tensor of type float16, return float16
    def encode_latents(self,latents_to_encode):
        return self.vae.encode(latents_to_encode).latent_dist.sample()* 0.18215



