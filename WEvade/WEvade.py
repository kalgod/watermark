import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import transform_image, project


def WEvade_W(xr, Decoder, criterion, args):
    watermarked_image=xr
    watermarked_image_cloned = xr.clone()
    r = args.rb
    lr = args.alpha
    epsilon = args.epsilon
    success = False

    # WEvade_W_II target watermark selection.
    if args.WEvade_type == 'WEvade-W-II':
        random_watermark = np.random.choice([0, 1], (xr.shape[0], args.watermark_length))
        target_watermark = torch.from_numpy(random_watermark).cuda().float()

    # WEvade_W_I target watermark selection.
    elif args.WEvade_type == 'WEvade-W-I':
        chosen_watermark = Decoder(watermarked_image).detach().cpu().numpy().round().clip(0, 1)
        chosen_watermark = 1 - chosen_watermark
        target_watermark = torch.from_numpy(chosen_watermark).cuda()

    for i in range(args.iteration):
        watermarked_image = watermarked_image.requires_grad_(True)
        min_value, max_value = torch.min(watermarked_image), torch.max(watermarked_image)
        decoded_watermark = Decoder(watermarked_image)

        # Post-process the watermarked image.
        loss = criterion(decoded_watermark, target_watermark)
        
        grads = torch.autograd.grad(loss, watermarked_image,create_graph=False)
        # with torch.no_grad():
        watermarked_image = watermarked_image - lr * grads[0]
        watermarked_image = torch.clamp(watermarked_image, min_value, max_value)

        # Projection.
        perturbation_norm = torch.norm(watermarked_image - watermarked_image_cloned, float('inf'))
        # print(i,loss,perturbation_norm)
        if perturbation_norm.cpu().detach().numpy() >= r:
            c = r / perturbation_norm
            watermarked_image = project(watermarked_image, watermarked_image_cloned, c)

        decoded_watermark = Decoder(watermarked_image)
        rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
        bit_acc_target = 1 - np.sum(np.abs(rounded_decoded_watermark - target_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)

        # Early Stopping.
        if perturbation_norm.cpu().detach().numpy() >= r:
            break

        if bit_acc_target >= 1 - epsilon:
            success = True
            break

    return watermarked_image

def WEvade_W_binary_search_r(xr, Decoder, criterion, args):

    watermarked_image=xr
    watermarked_image_cloned = xr.clone()

    rb = args.rb
    ra = 0
    lr = args.alpha
    epsilon = args.epsilon

    # WEvade_W_II target watermark selection.
    if args.WEvade_type == 'WEvade-W-II':
        random_watermark = np.random.choice([0, 1], (xr.shape[0], args.watermark_length))
        target_watermark = torch.from_numpy(random_watermark).cuda().float()

    # WEvade_W_I target watermark selection.
    elif args.WEvade_type == 'WEvade-W-I':
        chosen_watermark = Decoder(watermarked_image).detach().cpu().numpy().round().clip(0, 1)
        chosen_watermark = 1 - chosen_watermark
        target_watermark = torch.from_numpy(chosen_watermark).cuda()

    while (rb - ra >= 0.001):
        r = (rb + ra) / 2
        success = False

        for i in range(args.iteration):
            watermarked_image = watermarked_image.requires_grad_(True)
            min_value, max_value = torch.min(watermarked_image), torch.max(watermarked_image)
            decoded_watermark = Decoder(watermarked_image)

            loss = criterion(decoded_watermark, target_watermark)
            grads = torch.autograd.grad(loss, watermarked_image)
            with torch.no_grad():
                watermarked_image = watermarked_image - lr * grads[0]
                watermarked_image = torch.clamp(watermarked_image, min_value, max_value)

            perturbation_norm = torch.norm(watermarked_image - watermarked_image_cloned, float('inf'))
            if perturbation_norm.cpu().detach().numpy() >= r:
                c = r / perturbation_norm
                watermarked_image = project(watermarked_image, watermarked_image_cloned, c)

            decoded_watermark = Decoder(watermarked_image)
            rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
            bit_acc_target = 1 - np.sum(np.abs(rounded_decoded_watermark - target_watermark.cpu().numpy())) / (original_image.shape[0] * args.watermark_length)

            if perturbation_norm.cpu().detach().numpy() >= r:
                break

            if bit_acc_target >= 1 - epsilon:
                success = True
                break

        # Binary search
        if success:
            rb = r
        else:
            ra = r

    return watermarked_image


