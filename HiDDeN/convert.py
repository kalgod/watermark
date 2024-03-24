import os
import pprint
import argparse
import torch
import pickle
import utils
import logging
import sys

from options import *
from model.hidden import Hidden
from model.encoder_decoder import EncoderDecoder
from noise_layers.noiser import Noiser
from noise_argparser import NoiseArgParser

from train import train


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')
    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    # new_run_parser.add_argument('--data-dir', '-d', required=True, type=str,
    #                             help='The directory where the data is stored.')
    # new_run_parser.add_argument('--batch-size', '-b', required=True, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=300, type=int, help='Number of epochs to run the simulation.')
    # new_run_parser.add_argument('--name', required=True, type=str, help='The name of the experiment.')

    new_run_parser.add_argument('--size', '-s', default=128, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    # parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
    #                     help='If specified, use adds a Tensorboard log. On by default')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')

    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")

    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--folder', '-f', required=True, type=str,
                                 help='Continue from the last checkpoint in this folder.')
    continue_parser.add_argument('--data-dir', '-d', required=False, type=str,
                                 help='The directory where the data is stored. Specify a value only if you want to override the previous value.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int,
                                help='Number of epochs to run the simulation. Specify a value only if you want to override the previous value.')
    # continue_parser.add_argument('--tensorboard', action='store_true',
    #                             help='Override the previous setting regarding tensorboard logging.')

    args = parent_parser.parse_args()
    print(args)
    checkpoint = None
    loaded_checkpoint_file_name = None

    noise_config = []
    hidden_config = HiDDenConfiguration(H=args.size, W=args.size,
                                        message_length=args.message,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=8, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        decoder_loss=1,
                                        encoder_loss=0.7,
                                        adversarial_loss=1e-3,
                                        enable_fp16=args.enable_fp16
                                        )

    noiser = Noiser(noise_config, device)
    enc_dec = EncoderDecoder(hidden_config, noiser)

    filename="no-noise"
    checkpoint = torch.load("../stable_signature/hidden/ckpts/hidden_replicate.pth")
    print(checkpoint['encoder_decoder'].encoder,enc_dec.encoder)
    enc_dec.load_state_dict(checkpoint['encoder_decoder'])
    

    checkpoint = {
        'enc-model': enc_dec.encoder.state_dict(),
        'dec-model': enc_dec.decoder.state_dict(),
    }
    torch.save(checkpoint, "./converted/"+filename+".pth")


if __name__ == '__main__':
    main()
