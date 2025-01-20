
import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from torch import nn
from safetensors import safe_open

from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from safetensors.torch import load_file
import torch
from PIL import Image
import PIL
import torchvision.transforms as transforms

image = np.asarray(Image.open('image3.png'))

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

print(image.shape)




#vae = AutoencoderKL.from_pretrained("results/encoder_epoch_test")
vae = AutoencoderKL.from_single_file('R:/Projects/vae/create_dataset/sdxl_rec2100pq_v1.safetensors')
#vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

vae.eval()

has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
print("Has latents mean", has_latents_mean)
print("Has latents STD", has_latents_std)
print("scaling_factor", vae.config.scaling_factor)



#tensor = torch.zeros(1, 3, 512, 512)
tensor = torch.from_numpy(image.copy()).type('torch.FloatTensor').permute(2,0,1)[None, 0:3,:,:] / 255
print(tensor.shape)

tensor = tensor * 2 - 1.0

res = vae.encode(tensor).latent_dist.sample()# * vae.config.scaling_factor


print("Minimum value:", res.min().item(), "Maximum value:", res.max().item())

#save_image(rescale(res[:, 0, :, :].detach(), (-25.,25.), (0.,1.)), 'middle_0.png')
#save_image(rescale(res[:, 1, :, :].detach(), (-25.,25.), (0.,1.)), 'middle_1.png')
#save_image(rescale(res[:, 2, :, :].detach(), (-25.,25.), (0.,1.)), 'middle_2.png')
#save_image(rescale(res[:, 3, :, :].detach(), (-25.,25.), (0.,1.)), 'middle_3.png')

res = vae.decode(res).sample
#res = vae.decode(res)[0]
res = ((res + 1.0) / 2.0).clamp(0., 1.)

vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize =False)

save_image(res.detach(), 'out-base.png')

res = image_processor.postprocess(res.detach(), output_type = "pil") #output_type = output_type
print(res)
res[0].save("out.png")
#print(res)

#print("Minimum value:", res.min().item(), "Maximum value:", res.max().item())

#save_image(res.detach(), 'out.png')

#print(res.shape)

