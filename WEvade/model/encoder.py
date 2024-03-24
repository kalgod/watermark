import torch
import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu

from model.attenuations import JND
from torchvision import transforms
from PIL import Image   
NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
attenuation = JND(preprocess=UNNORMALIZE_IMAGENET).to(device)

class Encoder(nn.Module):
    ### Embed a watermark into the original image and output the watermarked image.
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.decoder_blocks=config.decoder_blocks
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels,decoder_blocks=self.decoder_blocks)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels,decoder_blocks=self.decoder_blocks)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.watermark_length, self.conv_channels,decoder_blocks=self.decoder_blocks)
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)
        self.tanh=nn.Tanh()

    def forward(self, original_image, watermark):
        # First, add two dummy dimensions in the end of the watermark.
        # This is required for the .expand to work correctly.
        watermark_transform=2*watermark-1
            
        if (self.decoder_blocks==8): expanded_watermark = watermark_transform.unsqueeze(-1)
        else: expanded_watermark = watermark.unsqueeze(-1)
        expanded_watermark.unsqueeze_(-1)
        expanded_watermark = expanded_watermark.expand(-1, -1, self.H, self.W)
        encoded_image = self.conv_layers(original_image)

        # Concatenate expanded watermark and the original image.
        concat = torch.cat([expanded_watermark, encoded_image, original_image], dim=1)
        watermarked_image = self.after_concat_layer(concat)
        watermarked_image = self.final_layer(watermarked_image)

        if (self.decoder_blocks==8):
            watermarked_image=self.tanh(watermarked_image)
            heatmaps=attenuation.heatmaps(original_image)
            watermarked_image=heatmaps*watermarked_image
            watermarked_image=original_image+1.5*watermarked_image

        return watermarked_image
