import os
from PIL import Image
import numpy as np

from diffusers import AutoencoderKL
from datasets import IterableDataset
from datasets import Dataset, Features, Value, Array3D
import torch
import datasets

from color_transforms import convert_srgb_to_rec2020_pq

import torch
from torch import nn
from torch.optim import AdamW

from tqdm import tqdm

tile_size = 720

def convert_to_rec2100(image):
    return convert_srgb_to_rec2020_pq(image.transpose((1,2,0))).transpose((2,0,1))
    
def preprocess(tensor):
    return tensor * 2 - 1.0
def postprocess(tensor):
    return (tensor + 1.0) / 2.0

def image_generator_720():
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").cuda()
    vae.eval()

    input_folder = "output_768"

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # Check if the file is an image
        try:
            with torch.no_grad():
                with Image.open(file_path) as img:
                    img_np = np.asarray(img.convert('RGB').resize((tile_size, tile_size), Image.LANCZOS)).astype(np.float32) / 255.0
                    
                    source = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).cuda()
                    
                    latent = vae.encode(preprocess(source)).latent_dist.sample()
                    
                    result = postprocess(vae.decode(latent).sample).clamp(0., 1.)
                    
                    
                    latent_np = latent.detach().squeeze().cpu().numpy()
                    source_np = (source.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
                    result_np = (result.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
                    
                    yield {
                        "input": source_np,
                        "latent": latent_np,
                        "output": result_np
                    }
                    
                    
        except Exception as e:
            print(f"Skipping file {file_name}: {e}")
        
            

# Create an iterable dataset
dataloader = Dataset.from_generator(
    image_generator_720,
    Features({
        "input" : Array3D(dtype='uint8'  , shape=(3, tile_size   , tile_size   )), # Full quality input (SDR)
        "latent": Array3D(dtype='float32', shape=(4, tile_size//8, tile_size//8)), # Latent encode with original SDXL VAE
        "output": Array3D(dtype='uint8'  , shape=(3, tile_size   , tile_size   ))  # Degraded result after decoding with original SDXL VAE
    })
)

#dataloader.save_to_disk("test_dataset3")



from collections import defaultdict
class RunningAverage():
    def __init__(self):
        self.values = defaultdict(float)
        self.size = 0.0
        self.count = 0.0
        
    def SetSize(self, value):
        self.size = value
        self.count += value
        
    def Add(self, key, value):
        self.values[key] += value.mean().item() * self.size
        
    def __getitem__(self, key):
        return self.values[key] / self.count




from torchvision.utils import save_image

vae = AutoencoderKL.from_single_file("R:/Projects/vae/create_dataset/trained_encoder.safetensors").cuda()
vae.train()

# Define the optimizer
optimizer = AdamW(vae.parameters(), lr=1e-4)

# Define loss function
loss_fn = nn.MSELoss()

num_epochs = 5
for epoch in range(num_epochs):
    stats = RunningAverage()
    train_bar = tqdm(dataloader)
    for batch in train_bar:
        input_np  = np.asarray(batch[ "input"]) / 255.0
        output_np = np.asarray(batch["output"]) / 255.0
        inputPq_np  = convert_to_rec2100( input_np)
        outputPq_np = convert_to_rec2100(output_np)
        
        input    = torch.from_numpy( input_np  ).type('torch.FloatTensor').unsqueeze(0).cuda()
        output   = torch.from_numpy(output_np  ).type('torch.FloatTensor').unsqueeze(0).cuda()
        inputPq  = torch.from_numpy( inputPq_np).type('torch.FloatTensor').unsqueeze(0).cuda()
        outputPq = torch.from_numpy(outputPq_np).type('torch.FloatTensor').unsqueeze(0).cuda()
        
        original_latent = torch.from_numpy(np.asarray(batch["latent"])).unsqueeze(0).type('torch.FloatTensor').cuda()
        
        
        optimizer.zero_grad()
        # Encode PQ to SDR latent
        latent_pq = vae.encode(preprocess(inputPq)).latent_dist.sample()
        loss_encoder = loss_fn(latent_pq / 25, original_latent / 25) # Divide by 25 to make the range -1 to 1.0, which is more similar to the image losses
        loss_encoder.backward()
        optimizer.step()
        
        optimizer.zero_grad()
        # Decode it back to the expected degraded PQ result
        result = postprocess(vae.decode(original_latent).sample)
        loss_decoder = loss_fn(result, outputPq)
        
        loss_decoder.backward()
        optimizer.step()
        
        #torch.cuda.empty_cache()
        # Encode and decode SDR (fake full range HDR)
        optimizer.zero_grad()
        result_hdr = postprocess(vae.decode(vae.encode(preprocess(input)).latent_dist.sample()).sample)
        loss_hdr = loss_fn(result_hdr, output)
        loss_hdr.backward()
        
        optimizer.step()
        #(loss_encoder + loss_hdr + loss_decoder).backward()
        
        stats.SetSize(result.size(0))
        stats.Add('mse_enc', loss_encoder)
        stats.Add('mse_dec', loss_decoder)
        stats.Add('mse_hdr', loss_hdr)
        train_bar.set_description(desc='[%d/%d] MSE enc: %.4f MSE dec: %.4f MSE hdr: %.4f' % (epoch, num_epochs, stats['mse_enc'], stats['mse_dec'], stats['mse_hdr']))
        #train_bar.set_description(desc='[%d/%d] MSE enc: %.4f MSE dec: %.4f' % (epoch, num_epochs, stats['mse_enc'], stats['mse_dec']))
    #print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    vae.save_pretrained(f"results/encoder_epoch_{epoch}")
    with torch.no_grad():
        input = torch.from_numpy(np.asarray(dataloader[0]["input"])).unsqueeze(0).type('torch.FloatTensor').cuda() / 255.0

        
        latent = vae.encode(input * 2 - 1.0).latent_dist.sample()
                    
        result = vae.decode(latent).sample
        
        #postprocess
        result = ((result + 1.0) / 2.0).clamp(0., 1.)
        if epoch == 0:
            save_image(input, f"epoch_{epoch}_input.png")
        save_image(result, f"epoch_{epoch}_result.png")
vae.save_pretrained(f"results/encoder_epoch_test")