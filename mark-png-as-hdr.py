#!/usr/bin/env python3

import sys
from PIL import Image, PngImagePlugin

# This is based on this GIST for CICP:
# https://gist.github.com/zhuowei/96b6e184bcf2de64433fbb86e15f7762
# which is a hack around not being able to add it directly. I have opened an issue to fix this in PIL:
# https://github.com/python-pillow/Pillow/issues/8703
#
# SD metadata is based on stable-diffusion-webui code

def putchunk_hook(fp, cid, *data):
    if cid == b"haxx":
        cid = b"cICP"
    return PngImagePlugin.putchunk(fp, cid, *data)


with Image.open(sys.argv[1]) as im:
    pnginfo = PngImagePlugin.PngInfo()
    
    # Add existing metadata back
    for key, value in im.info.items():
        if isinstance(key, str) and isinstance(value, str):
            pnginfo.add_text(key, value)
    
    pnginfo.add(b"haxx", bytes([9, 16, 0, 1]))
    im.encoderinfo = {"pnginfo": pnginfo}
    with open(sys.argv[2], "wb") as outfile:
        PngImagePlugin._save(im, outfile, sys.argv[2], chunk=putchunk_hook)