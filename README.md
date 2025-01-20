Rec.2100 vae training
=====

WIP training script for fine-tuning the Stable Diffusion XL VAE to output Rec. 2100 PQ instead of the normal sRGB.
They are not intended for general use and will contain hardcoded paths, etc.

The idea is to match the existing SDR latent to the corresponding Rec. 2100 PQ version, such that the image doesn't change when you switch the VAE, only the color space.

The training script `train_vae_encode.py` tries to optimize the following:

- Encode sRGB input encoded as Rec. 2100 PQ to the sRGB latent provided by SDXL VAE
- Decode sRGB latent from SDXL encode to the decoded output of SDXL VAE converted to Rec. 2100 PQ
- Encode full-range Rec. 2100 PQ input and decode again to match expected decoded output.

The loss is a simple L2 loss and the final result is a little less sharp than the original, but I have not looked into what is good loss functions for this and this was a quick and dirty attempt.