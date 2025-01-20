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

tile_size = 768

def image_generator():
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").cuda()
    vae.eval()

    input_folder = "output_768"

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # Check if the file is an image
        try:
            with torch.no_grad():
                with Image.open(file_path) as img:
                    #print("img_in", np.asarray(img).shape)
                    source = torch.from_numpy(np.asarray(img.convert('RGB')).copy()).type('torch.FloatTensor').permute(2,0,1)[None, 0:3,:,:].cuda() / 255.0
                    #print("img_converted", source.shape)
                    #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                    
                    #preprocess
                    source = source * 2 - 1.0
                    
                    latent = vae.encode(source).latent_dist.sample()
                    
                    result = vae.decode(latent).sample
                    
                    #postprocess
                    result = ((result + 1.0) / 2.0).clamp(0., 1.)
                    
                    latent_np = latent.detach().squeeze().cpu().numpy()
                    source_np = (convert_srgb_to_rec2020_pq(source.detach().cpu().numpy()[0, :, :, :].transpose((1,2,0))) * 255).astype(np.uint8).transpose((2,0,1))
                    result_np = (convert_srgb_to_rec2020_pq(result.detach().cpu().numpy()[0, :, :, :].transpose((1,2,0))) * 255).astype(np.uint8).transpose((2,0,1))
                    
                    yield {"input": source_np, "latent": latent_np, "result": result_np}
                    
                    
        except Exception as e:
            print(f"Skipping file {file_name}: {e}")
        
            

# Create an iterable dataset
dataloader = Dataset.from_generator(
    image_generator,
    Features({
        "input": Array3D(dtype='uint8', shape=(3, tile_size, tile_size)),
        "latent": Array3D(dtype='float32', shape=(4, tile_size//8, tile_size//8)),
        "result": Array3D(dtype='uint8', shape=(3, tile_size, tile_size))
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





vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").cuda()
vae.train()

# Define the optimizer
optimizer = AdamW(vae.parameters(), lr=1e-4)

# Define loss function
loss_fn = nn.MSELoss()


#from torchvision.utils import save_image
#for batch in dataloader:
#    expected = torch.from_numpy(np.asarray(batch["result"])).unsqueeze(0).type('torch.FloatTensor').cuda() / 255.0
#    latent = torch.from_numpy(np.asarray(batch["latent"])).unsqueeze(0).type('torch.FloatTensor').cuda()
#    result = vae.decode(latent).sample
#    result = ((result + 1.0) / 2.0).clamp(0., 1.)
#    
#    save_image(expected.detach(), 'dataset-expected.png')
#    save_image(result.detach(), 'dataset-result.png')
#    
#    break

num_epochs = 1
for epoch in range(num_epochs):
    stats = RunningAverage()
    train_bar = tqdm(dataloader)
    for batch in train_bar:
        expected = torch.from_numpy(np.asarray(batch["result"])).unsqueeze(0).type('torch.FloatTensor').cuda() / 255.0
        latent = torch.from_numpy(np.asarray(batch["latent"])).unsqueeze(0).type('torch.FloatTensor').cuda()
        result = vae.decode(latent).sample
        result = ((result + 1.0) / 2.0)#.clamp(0., 1.)
        
        loss = loss_fn(result, expected)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        stats.SetSize(expected.size(0))
        stats.Add('mse', loss)
        train_bar.set_description(desc='[%d/%d] MSE: %.4f' % (epoch, num_epochs, stats['mse']))
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    vae.save_pretrained(f"results/decoder_epoch_{epoch}")