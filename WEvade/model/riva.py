import torch
import torch.nn as nn
from PIL import Image
import cv2
import os
from imwatermark import WatermarkEncoder, WatermarkDecoder
from torchvision import transforms
import subprocess
from utils import *

class Watermarker:
    def encode(self, img_path, output_path, prompt=''):
        raise NotImplementedError

    def decode(self, img_path):
        raise NotImplementedError

class InvisibleWatermarker(Watermarker):
    def __init__(self, wm_text, method):
        if method == 'rivaGan':
            WatermarkEncoder.loadModel()
        self.method = method
        encoder = WatermarkEncoder()
        self.wm_type = 'bytes'
        self.wm_text = wm_text
        encoder.set_watermark(self.wm_type, self.wm_text.encode('utf-8'))
        self.encoder=WatermarkEncoderModel(encoder,method)
        decoder = WatermarkDecoder(self.wm_type, len(self.wm_text) * 8)
        self.decoder=WatermarkDecoderModel(decoder,method)
    
class WatermarkEncoderModel(nn.Module):
    def __init__(self, watermark_encoder,method):
        super(WatermarkEncoderModel, self).__init__()
        self.watermark_encoder = watermark_encoder
        self.method = method

    def forward(self, x0):
        tensor = x0.clone()
        tensor = (tensor + 1) / 2
        tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        res = x0.clone()
        for i in range(tensor.shape[0]):
            out = self.watermark_encoder.encode(tensor[i], self.method)
            out = torch.from_numpy(out).permute(2, 0, 1).to(device).float()
            out = out / 255
            out = 2 * out - 1
            res[i] = out
        return res

class WatermarkDecoderModel(nn.Module):
    def __init__(self, watermark_decoder,method):
        super(WatermarkDecoderModel, self).__init__()
        self.watermark_decoder = watermark_decoder
        self.method = method

    def forward(self, xr):
        tensor = xr.clone()
        tensor = (tensor + 1) / 2
        tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

        res = torch.zeros((xr.shape[0], 32)).to(device).float()
        for i in range(tensor.shape[0]):
            wm_text = self.watermark_decoder.decode(tensor[i], self.method)
            try:
                if type(wm_text) == bytes:
                    a = bytearray_to_bits('test'.encode('utf-8'))
                    b = bytearray_to_bits(wm_text)
                elif type(wm_text) == str:
                    a = bytearray_to_bits('test'.encode('utf-8'))
                    b = bytearray_to_bits(wm_text.encode('utf-8'))
            except:
                print(f'failed to decode {wm_text}', type(wm_text), len(wm_text))
                pass
            b = torch.Tensor(np.array(b)).to(device)
            res[i] = b
        return res