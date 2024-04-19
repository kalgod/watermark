import torch
import torch.nn as nn
from PIL import Image
import cv2
import os
from imwatermark import WatermarkEncoder, WatermarkDecoder
from torchvision import transforms
import subprocess
from utils import *

class CIN_Watermarker:
    def encode(self, img_path, output_path, prompt=''):
        raise NotImplementedError

    def decode(self, img_path):
        raise NotImplementedError

class CIN(CIN_Watermarker):
    def __init__(self, cinnet):
        self.encoder=CIN_WatermarkEncoderModel(cinnet.module)
        self.decoder=CIN_WatermarkDecoderModel(cinnet.module)
    
class CIN_WatermarkEncoderModel(nn.Module):
    def __init__(self, cinnet):
        super().__init__()
        self.invertible_model = cinnet.invertible_model
        self.cs_model = cinnet.cs_model
        self.fusion_model= cinnet.fusion_model
        self.invDown = cinnet.invDown
        self.decoder2 = cinnet.decoder2
        self.nsm_model = cinnet.nsm_model

    def forward(self, x0, msg):
        image=x0.clone()
        # down                                             #[128]
        cover_down = self.invDown(image)                #[64]
        # fusion
        fusion = self.fusion_model(cover_down, msg, self.invDown)          #[64]
        # inv_forward
        inv_encoded = self.invertible_model(fusion)    #[64]
        # cs
        cs = self.cs_model(inv_encoded, cover_down)
        # up to out
        watermarking_img = self.invDown(cs, rev=True).clamp(-1, 1)   #[128]
        return watermarking_img

class CIN_WatermarkDecoderModel(nn.Module):
    def __init__(self, cinnet):
        super().__init__()
        self.invertible_model = cinnet.invertible_model
        self.cs_model = cinnet.cs_model
        self.fusion_model= cinnet.fusion_model
        self.invDown = cinnet.invDown
        self.decoder2 = cinnet.decoder2
        self.nsm_model = cinnet.nsm_model

    def nsm(self, noised_img):
        return torch.round(torch.mean((torch.argmax(self.nsm_model(noised_img.clone().detach().clamp(-1,1)), dim=1)).float()))

    def forward(self, xr):
        noised_img = xr.clone()
        pre_noise = self.nsm(noised_img)
        if pre_noise == 1:
            # decoder1
            msg_fake_1 = None
            img_fake = torch.zeros_like(noised_img).cuda()
            # decoder2 
            msg_fake_2 = self.decoder2(noised_img)#.clamp(-1, 1)
        else:
            # decoder1                                           
            down = self.invDown(noised_img)            #[64]
            cs_rev = self.cs_model(down, rev=True)
            inv_back = self.invertible_model(cs_rev, rev=True)   #[64]
            img_fake, msg_fake_1 = self.fusion_model(inv_back, None, self.invDown, rev=True)   #[64]
            img_fake = self.invDown(img_fake, rev=True)   #[128]
            img_fake = img_fake.clamp(-1, 1)
            # decoder2
            msg_fake_2 = None
        #
        msg_nsm = msg_fake_1 if msg_fake_1 is not None else msg_fake_2
        msg_nsm=msg_nsm/max(1,0.66*torch.max(msg_nsm))
        # print(msg_nsm)
        return msg_nsm