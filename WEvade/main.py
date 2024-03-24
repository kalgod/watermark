import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from tqdm import tqdm
from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from utils import *
from model.model import Model
from WEvade import WEvade_W, WEvade_W_binary_search_r
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.brightness import Brightness

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_fn = lpips.LPIPS(net='vgg')  # 或者 'alex' 也是可选的
loss_fn = loss_fn.to(device)

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

def post_process(original_image, Encoder, Decoder, method, args):
    # Embed the ground-truth watermark into the original image.
    original_image = original_image.cuda()
    groundtruth_watermark = torch.from_numpy(np.load('./watermark/watermark_coco.npy')).cuda().repeat(original_image.shape[0],1)
    watermarked_image = Encoder(original_image, groundtruth_watermark)
    watermarked_image = transform_image(watermarked_image)
    watermarked_image_cloned = watermarked_image.clone()

    if method == 'jpeg':
        noise_layer = DiffJPEG(args.Q)
        watermarked_image = noise_layer(watermarked_image)
    elif method == 'gaussian':
        noise_layer = Gaussian(args.sigma1)
        watermarked_image = noise_layer(watermarked_image)
    elif method == 'gaussianblur':
        noise_layer = GaussianBlur(args.sigma2)
        watermarked_image = noise_layer(watermarked_image)
    elif method == 'brightness':
        noise_layer = Brightness(args.a)
        watermarked_image = noise_layer(watermarked_image)
    else:
        pass

    post_processed_watermarked_image = watermarked_image
    decoded_watermark = Decoder(post_processed_watermarked_image)
    rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
    bound = torch.norm(post_processed_watermarked_image - watermarked_image_cloned, float('inf'))
    bit_acc_groundtruth = 1 - np.sum(np.abs(rounded_decoded_watermark - groundtruth_watermark.cpu().numpy())) / (original_image.shape[0] * args.watermark_length)
    return bit_acc_groundtruth, bound.item(),post_processed_watermarked_image,watermarked_image_cloned

def main():
    parser = argparse.ArgumentParser(description='WEvade-W Arguments.')
    parser.add_argument('--checkpoint', default='./converted/clean.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='./dataset/coco/val', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=256, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--batch', default=10, type=int, help='batch size.')
    parser.add_argument('--tau', default=0.8, type=float, help='Detection threshold of the detector.')
    parser.add_argument('--iteration', default=100, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--epsilon', default=0.01, type=float, help='Epsilon used in WEvdae-W.')
    parser.add_argument('--alpha', default=1, type=float, help='Learning rate used in WEvade-W.')
    parser.add_argument('--rb', default=2, type=float, help='Upper bound of perturbation.')
    parser.add_argument('--WEvade-type', default='WEvade-W-II', type=str, help='Using WEvade-W-I/II.')
    parser.add_argument('--detector-type', default='double-tailed', type=str, help='Using double-tailed/single-tailed detctor.')
    # In our algorithm, we use binary-search to obtain perturbation upper bound. But in the experiment, we find
    # binary-search actually has no significant effect on the perturbation results. And we reduce time cost if not
    # using binary-search.
    parser.add_argument('--binary-search', default=False, type=bool, help='Whether use binary-search to find perturbation.')

    parser.add_argument('--Q', default=50, type=int, help='Parameter Q for JPEGCompression.')
    parser.add_argument('--sigma1', default=0.1, type=float, help='Parameter \sigma for Gaussian noise.')
    parser.add_argument('--sigma2', default=0.5, type=float, help='Parameter \sigma for Gaussian blur.')
    parser.add_argument('--a', default=1.5, type=float, help='Parameter a for Brightness/Contrast.')

    args = parser.parse_args()

    # Load model.
    model = Model(args.image_size, args.watermark_length, device)
    checkpoint = torch.load(args.checkpoint)

    model.encoder.load_state_dict(checkpoint['enc-model'])
    model.decoder.load_state_dict(checkpoint['dec-model'])

    # Load dataset.
    data = get_data_loaders(args.image_size, args.dataset_folder,args.batch)
    # WEvade.
    start_time = time.time()
    model.encoder.eval()
    model.decoder.eval()
    criterion = nn.MSELoss().cuda()

    Bit_acc = AverageMeter()
    Perturbation = AverageMeter()
    Evasion_rate = AverageMeter()
    Batch_time = AverageMeter()

    method_list = ['jpeg', 'gaussian', 'gaussianblur', 'brightness']
    method_list = ['wevade']
    for method in method_list:
        folder="./result/"+method
        os.makedirs(folder, exist_ok=True)
        print('method:', method)
        for idx, (image, _) in tqdm(enumerate(data), total=len(data)):
            image = transform_image(image).to(device)

            if (method!='wevade'):
                bit_acc, bound,image_attack,image_wm= post_process(image, model.encoder, model.decoder, method, args)
            else:
                # Repeat up to three times to prevent randomly picked target watermark from being bad.
                for k in range(3):
                    if args.binary_search == False:
                        bit_acc, bound, success,image_attack,image_wm = WEvade_W(image, model.encoder, model.decoder, criterion, args)
                    else:
                        bit_acc, bound, success,image_attack,image_wm = WEvade_W_binary_search_r(image, model.encoder, model.decoder, criterion, args)
                    if success:
                        break
            
            save_images(image,image_wm,image_attack,folder+"/"+str(idx)+".png")
            # Detection for double-tailed/single-tailed detector.
            if args.detector_type == 'double-tailed':
                evasion = (1-args.tau <= bit_acc and bit_acc <= args.tau)
            elif args.detector_type == 'single-tailed':
                evasion = (bit_acc <= args.tau)

            image=np.asarray(image.detach().cpu().numpy())
            image_wm=np.asarray(image_wm.detach().cpu().numpy())
            image_attack=np.asarray(image_attack.detach().cpu().numpy())
            psnr=peak_signal_noise_ratio(image,image_wm)
            ssim=structural_similarity(image,image_wm,channel_axis=1,data_range=2.0)
            lpips_score=cal_lpips(image,image_wm)
            print("For watermark, PSNR:",psnr,"SSIM:",ssim,"LPIPS:",lpips_score)
            psnr=peak_signal_noise_ratio(image,image_attack)
            ssim=structural_similarity(image,image_attack,channel_axis=1,data_range=2.0)
            lpips_score=cal_lpips(image,image_attack)
            print("For attacks, PSNR:",psnr,"SSIM:",ssim,"LPIPS:",lpips_score,"ACC:",bit_acc)
            bound = bound / 2   # [-1,1]->[0,1]
            Bit_acc.update(bit_acc, image.shape[0])
            Perturbation.update(bound, image.shape[0])
            Evasion_rate.update(evasion, image.shape[0])
            Batch_time.update(time.time() - start_time)
            start_time = time.time()

        print("Average Bit_acc=%.4f\t Average Perturbation=%.4f\t Evasion rate=%.2f\t Time=%.2f" % (Bit_acc.avg, Perturbation.avg, Evasion_rate.avg, Batch_time.sum))


if __name__ == '__main__':
    main()