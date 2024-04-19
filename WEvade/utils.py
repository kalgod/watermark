import torch
from torchvision import datasets, transforms
import torchvision.utils
import os
from PIL import Image
from augly.image import functional as aug_functional
import numpy as np
from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import lpips

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_fn = lpips.LPIPS(net='vgg')  # 或者 'alex' 也是可选的
loss_fn = loss_fn.to(device)

def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def cal_fid(img1,img2):
    # 初始化预训练的Inception模型
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    mu_real, sigma_real = compute_statistics_of_path(img1, model, 50)  # 这里使用50张真实图像
    mu_fake, sigma_fake = compute_statistics_of_path(img2, model, 50)  # 这里使用50张生成图像
    # 计算FID
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score

def cal_lpips(img1,img2):
    # 初始化lpips模型
    img1=torch.from_numpy(img1).float()
    img2=torch.from_numpy(img2).float()
    img1 = img1.to(device)
    img2 = img2.to(device)
    lpips_score=loss_fn(img1,img2).detach().cpu().numpy()
    lpips_score=np.mean(lpips_score)
    # print(lpips_score.shape)
    return lpips_score

def cal_psnr(img1,img2):
    x=np.asarray(img1)
    y=np.asarray(img2)
    psnr=peak_signal_noise_ratio(x,y)
    return psnr

def cal_ssim(img1,img2):
    x=np.asarray(img1)
    x=np.tile(x, (2, 1, 1, 1))
    x=np.tile(x, (2, 1, 1, 1))
    x=np.tile(x, (2, 1, 1, 1))
    y=np.asarray(img2)
    y=np.tile(y, (2, 1, 1, 1))
    y=np.tile(y, (2, 1, 1, 1))
    y=np.tile(y, (2, 1, 1, 1))
    ssim=structural_similarity(x,y,channel_axis=1,data_range=2.0)
    return ssim

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: Tensor image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    x = (x+1)/2
    for ii,img in enumerate(x):
        pil_img = to_pil(img)
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return img_aug*2-1

def save_images(original_images, watermarked_images,attack_images, folder,num=3):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
    attack_images = attack_images[:attack_images.shape[0], :, :, :].cpu()
    
    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2
    attack_images = (attack_images + 1) / 2

    if (num==3): stacked_images = torch.cat([images, watermarked_images,attack_images], dim=0)
    elif (num==2): 
        abs_watermarked_images=np.abs(watermarked_images-images)*10
        stacked_images = torch.cat([watermarked_images, abs_watermarked_images], dim=0)
    else:
        stacked_images = torch.cat([images], dim=0)
    torchvision.utils.save_image(stacked_images, folder,nrow=int(original_images.shape[0]))

def save_image_from_tensor(tensor, file_path):
    # Save a single image from torch tensor
    # refer to https://pytorch.org/vision/stable/_modules/torchvision/utils.html
    tensor = (tensor + 1) / 2   # for HiDDeN watermarking method only
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(tensor)
    im.save(file_path)


def get_data_loaders(image_size, dataset_folder,batch):
    # Get torch data loaders. The data loaders take a crop of the image, transform it into tensor, and normalize it.
    data_transforms = transforms.Compose([
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_images = datasets.ImageFolder(dataset_folder, data_transforms)
    dataset_loader = torch.utils.data.DataLoader(dataset_images, batch_size=batch, shuffle=False, num_workers=4)

    return dataset_loader


def transform_image(image):
    # For HiDDeN watermarking method, image pixel value range should be [-1, 1]. Transform an image into [-1, 1] range.
    cloned_encoded_images = (image + 1) / 2  # for HiDDeN watermarking method only
    cloned_encoded_images = cloned_encoded_images.mul(255).clamp_(0, 255)

    cloned_encoded_images = cloned_encoded_images / 255
    cloned_encoded_images = cloned_encoded_images * 2 - 1  # for HiDDeN watermarking method only
    image = cloned_encoded_images.cuda()

    return image


def project(param_data, backup, epsilon):
    # If the perturbation exceeds the upper bound, project it back.
    r = param_data - backup
    r = epsilon * r

    return backup + r


class AverageMeter(object):
    # Computes and stores the average and current value.
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count