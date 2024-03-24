import torch.nn as nn

class ConvBNRelu(nn.Module):
    # A block of Convolution, Batch Normalization, and ReLU activation
    def __init__(self, channels_in, channels_out, stride=1,decoder_blocks=7):

        super(ConvBNRelu, self).__init__()
        
        if (decoder_blocks==8):
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
                nn.BatchNorm2d(channels_out, eps=1e-3),
                nn.GELU()
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.layers(x)
