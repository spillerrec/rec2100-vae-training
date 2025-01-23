#!/usr/bin/env python3

import sys
from safetensors.torch import save_file, load_file, safe_open
import pprint

hdr_metadata = {
    'modelspec.sai_model_spec': '1.0.0',
    'modelspec.architecture': 'stable-diffusion-xl-v1-base',
    'modelspec.implementation' : 'https://github.com/Stability-AI/generative-models',
    'modelspec.title' : 'SDXL Rec. 2100 PQ VAE version 1.0',
    'modelspec.description' : 'This is a replacement VAE for SDXL which retains the original latent space, but takes Rec. 2100 as input and output',
    'modelspec.author' : 'spillerrec',
    'modelspec.date' : '2025-01-17T16:12:39',
    'modelspec.color_space' : 'cicp:9,16,0,true'
}

with safe_open(sys.argv[1], framework="pt") as f:
    tensors = {key: f.get_tensor(key) for key in f.keys()}
    metadata = f.metadata()
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(metadata)
    
    # Currently the VAE converted through ComfyUI contains the prompt metaddata which is pretty useless, so let us just completely replace it
    metadata = hdr_metadata
    
    if len(sys.argv) == 3:
        save_file(tensors, sys.argv[2], metadata=metadata)
    

