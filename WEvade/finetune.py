import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from tqdm import tqdm
from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import lpips
from diffusers import ReSDPipeline

from utils import *
from model.model import Model
from WEvade import WEvade_W, WEvade_W_binary_search_r
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.brightness import Brightness
from noise_layers.wmattacker import *

np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_fn = lpips.LPIPS(net='vgg')  # 或者 'alex' 也是可选的
loss_fn = loss_fn.to(device)

pipe = ReSDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.set_progress_bar_config(disable=True)
pipe.to(device)
print('Finished loading model')

attackers = {
    'diff_attacker_60': DiffWMAttacker(pipe, batch_size=5, noise_step=6, captions={}),
    'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
    'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
    'jpeg_attacker_50': JPEGAttacker(quality=50),
}

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
    y=np.asarray(img2)
    ssim=structural_similarity(x,y,channel_axis=1,data_range=2.0)
    return ssim

def white_adv(original_image,image,real_watermark, Decoder, args):
    original_image_cloned=original_image.clone()
    watermarked_image=image.clone()
    lr = args.alpha
    criterion = nn.MSELoss()

    # target_watermark = torch.Tensor(np.random.choice([0, 1], (image.shape[0], args.watermark_length))).to(device)
    target_watermark=real_watermark
    for i in range(args.iteration):
        watermarked_image = watermarked_image.requires_grad_(True)
        decoded_watermark = Decoder(watermarked_image)
        decoded_watermark=torch.clamp(decoded_watermark,0,1)
        acc_adv = 1 - np.sum(np.abs(decoded_watermark.detach().cpu().numpy().round() - real_watermark.cpu().numpy())) / (image.shape[0] * args.watermark_length)

        # Post-process the watermarked image.
        loss = abs(0.5-criterion(decoded_watermark, target_watermark))
        grads = torch.autograd.grad(loss, watermarked_image,create_graph=False)
        # with torch.no_grad():
        watermarked_image = watermarked_image - lr * grads[0]
        # print(i,loss.item(),acc_adv)
        # Projection.
        delta=watermarked_image - original_image_cloned
        delta=torch.clamp(delta,-2*args.delta_finetune,2*args.delta_finetune)
        bound = cal_psnr(original_image_cloned.detach().cpu().numpy(),watermarked_image.detach().cpu().numpy())
        # print(i,loss,bound)
        watermarked_image=original_image_cloned+delta
        # watermarked_image=watermarked_image.detach()
        
        # decoded_watermark = Decoder(watermarked_image)
        # rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
        # bit_acc_target = 1 - np.sum(np.abs(rounded_decoded_watermark - real_watermark.cpu().numpy())) / (original_image.shape[0] * args.watermark_length)

        # if bit_acc_target < args.tau:
        #     success = True
        #     break
    return watermarked_image

def apply_attack(x0,xr,real_watermark, Decoder, method,criterion, args,eval=True):
    # Embed the ground-truth watermark into the original image.
    image=xr.clone()
    if method == 'jpeg':
        noise_layer = DiffJPEG(args.Q)
        image = noise_layer(image)
        # image=jpeg_compress(image,args.Q)
    elif method == 'gaussian':
        noise_layer = Gaussian(args.sigma1)
        image = noise_layer(image)
    elif method == 'gaussianblur':
        noise_layer = GaussianBlur(args.sigma2)
        image = noise_layer(image)
    elif method == 'brightness':
        noise_layer = Brightness(args.a)
        image = noise_layer(image)
    elif method == 'white':
        image = white_adv(x0,xr,real_watermark, Decoder, args)
    elif method == 'wevade':
        image = WEvade_W(xr, Decoder, criterion, args)
    elif method == 'regen-cheng':
        name='cheng2020-anchor_3'
        noise_layer=attackers[name]
        image=(image+1)/2
        image = noise_layer.attack(image,eval=eval)
        image=transform_image(image)
    elif method == 'regen-bmsh':
        name='bmshj2018-factorized_3'
        noise_layer=attackers[name]
        image=(image+1)/2
        image = noise_layer.attack(image,eval=eval)
        image=transform_image(image)
    elif method == 'regen-diff':
        name='diff_attacker_60'
        noise_layer=attackers[name]
        image = noise_layer.attack(image,eval=eval)
        image=2*image-1
        image=transform_image(image)
    elif method == 'combined':
        noise_layer = DiffJPEG(args.Q)
        image = noise_layer(image)
        noise_layer = Gaussian(args.sigma1)
        image = noise_layer(image)
        noise_layer = GaussianBlur(args.sigma2)
        image = noise_layer(image)
        noise_layer = Brightness(args.a)
        image = noise_layer(image)
    else:
        pass

    return image

def direct_optimize(model,x0,real_watermark,criterion,args):
    model.encoder.eval()
    model.decoder.eval()
    
    tmp=model.encoder(x0,real_watermark)
    tmp=transform_image(tmp)
    xr=tmp.clone().detach().requires_grad_(True)
    params = [
        # {'params': model.encoder.parameters(), 'lr': args.lr_encoder},
        {'params': model.decoder.parameters(), 'lr': args.lr_decoder}  # 设置decoder参数的学习率为 1e-4
    ]

    # 创建优化器时使用 params 参数组设置不同的学习率
    optimizer = torch.optim.Adam(params)
    acc_adv=0

    def cal_xra(x0,xr,method):
        xra=apply_attack(x0,xr.clone(),real_watermark, model.decoder,method,criterion, args,eval=False)
        decoded_watermark_adv = model.decoder(xra)
        decoded_watermark_adv=torch.clamp(decoded_watermark_adv,0,1)
        acc_adv = 1 - np.sum(np.abs(decoded_watermark_adv.detach().cpu().numpy().round() - real_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)
        psnr_adv=cal_psnr(x0.detach().cpu().numpy(),xra.detach().cpu().numpy())
        loss=criterion(decoded_watermark_adv,real_watermark)
        return loss,acc_adv,psnr_adv

    for i in range(args.iter_finetune):
        xr = xr.requires_grad_(True)
        decoded_watermark_clean = model.decoder(xr)
        decoded_watermark_clean=torch.clamp(decoded_watermark_clean,0,1)
        acc_clean = 1 - np.sum(np.abs(decoded_watermark_clean.detach().cpu().numpy().round() - real_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)
        psnr_clean=cal_psnr(x0.detach().cpu().numpy(),xr.detach().cpu().numpy())

        loss_a1,acc_adv1,psnr_adv1=cal_xra(x0,xr,args.attack_train)
        loss_a2,acc_adv2,psnr_adv2=cal_xra(x0,xr,args.attack_train1)
        loss_a=loss_a1+loss_a2;cnt=2
        # loss_a3,acc_adv3,psnr_adv3=cal_xra(x0,xr,"regen-diff");loss_a+=loss_a3;cnt+=1
        # loss_a4,acc_adv4,psnr_adv4=cal_xra(x0,xr,"regen-bmsh");loss_a+=loss_a3;cnt+=1
        # loss_a5,acc_adv5,psnr_adv5=cal_xra(x0,xr,"regen-diff");loss_a+=loss_a5;cnt+=1

        loss_a=loss_a/cnt
        loss_w = criterion(decoded_watermark_clean, real_watermark)
        loss_i1=criterion(x0,xr)
        loss_i2=criterion(tmp,xr)
        loss_i=(1*loss_i1+1*loss_i2)/3

        loss=loss_a+0.1*loss_w+args.lamda_i*loss_i

        grads = torch.autograd.grad(loss, xr)
        xr = xr - args.lr_image * torch.sign(grads[0])
        delta=xr - x0
        delta=torch.clamp(delta,-2*args.delta_finetune,2*args.delta_finetune)
        # print(i,loss,bound)
        xr=x0+delta
        xr = transform_image(xr)
        xr=xr.detach()
        if (cnt==2):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2)
            if (acc_adv1>args.tau and acc_adv2>args.tau): break
        elif (cnt==3):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3)
            if (acc_adv1>args.tau and acc_adv2>args.tau and acc_adv3>args.tau): break
        elif (cnt==4):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3,acc_adv4)
            if (acc_adv1>args.tau and acc_adv2>args.tau and acc_adv3>args.tau and acc_adv4>args.tau): break
        elif (cnt==5):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3,acc_adv4,acc_adv5)
            if (acc_adv1>args.tau and acc_adv2>args.tau and acc_adv3>args.tau and acc_adv4>args.tau and acc_adv5>args.tau): break
    model.encoder.eval()
    model.decoder.eval()
    return xr

def update_model(model,x0,real_watermark,criterion,args,select):
    if (select=="encoder"):
        model.encoder.train()
        model.decoder.eval()
    else:
        model.encoder.eval()
        model.decoder.train()

    # model.encoder.train()
    # model.decoder.train()

    params_en = [
        {'params': model.encoder.parameters(), 'lr': args.lr_encoder}
    ]
    # 创建优化器时使用 params 参数组设置不同的学习率
    optimizer_en = torch.optim.Adam(params_en)

    params_de = [
        {'params': model.decoder.parameters(), 'lr': args.lr_decoder}
    ]
    # 创建优化器时使用 params 参数组设置不同的学习率
    optimizer_de = torch.optim.Adam(params_de)
    def cal_xra(x0,xr,method):
        xra=apply_attack(x0,xr.clone(),real_watermark, model.decoder,method,criterion, args,eval=False)
        decoded_watermark_adv = model.decoder(xra)
        decoded_watermark_adv=torch.clamp(decoded_watermark_adv,0,1)
        acc_adv = 1 - np.sum(np.abs(decoded_watermark_adv.detach().cpu().numpy().round() - real_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)
        psnr_adv=cal_psnr(x0.detach().cpu().numpy(),xra.detach().cpu().numpy())
        loss=criterion(decoded_watermark_adv,real_watermark)
        return loss,acc_adv,psnr_adv

    for i in range(args.iter_finetune):
        if (i==args.iter_finetune-1 and acc_adv1<=args.tau): 
            model.encoder.train()
            model.decoder.train()
        else:
            model.encoder.train()
            model.decoder.train()
        xr=model.encoder(x0,real_watermark)
        xr = transform_image(xr)

        decoded_watermark_clean = model.decoder(xr)
        decoded_watermark_clean=torch.clamp(decoded_watermark_clean,0,1)
        acc_clean = 1 - np.sum(np.abs(decoded_watermark_clean.detach().cpu().numpy().round() - real_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)
        psnr_clean=cal_psnr(x0.detach().cpu().numpy(),xr.detach().cpu().numpy())

        loss_a1,acc_adv1,psnr_adv1=cal_xra(x0,xr,args.attack_train)
        loss_a=loss_a1;cnt=1
        loss_a2,acc_adv2,psnr_adv2=cal_xra(x0,xr,args.attack_train1);loss_a+=loss_a2;cnt+=1
        # loss_a3,acc_adv3,psnr_adv3=cal_xra(x0,xr,"regen-diff");loss_a+=loss_a3;cnt+=1
        # loss_a4,acc_adv4,psnr_adv4=cal_xra(x0,xr,"regen-bmsh");loss_a+=loss_a3;cnt+=1
        # loss_a5,acc_adv5,psnr_adv5=cal_xra(x0,xr,"regen-diff");loss_a+=loss_a5;cnt+=1
        loss_a=loss_a/cnt
        
        loss_w = criterion(decoded_watermark_clean, real_watermark)
        # loss_i=criterion(x0,xr)
        loss_i=0.5*torch.mean(loss_fn(x0,xr))+0.5*criterion(x0,xr)
        loss=loss_a+0.1*loss_w+args.lamda_i*loss_i
        
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
        loss.backward()

        if (i==args.iter_finetune-1 and acc_adv1<=args.tau): optimizer_de.step()
        else: optimizer_en.step()

        # optimizer_en.step()
        # optimizer_de.step()
        if (cnt==1):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1)
            if (acc_adv1>args.tau): break
        if (cnt==2):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2)
            if (acc_adv1>args.tau and acc_adv2>args.tau): break
        elif (cnt==3):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3)
            if (acc_adv1>args.tau and acc_adv2>args.tau and acc_adv3>args.tau): break
        elif (cnt==4):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3,acc_adv4)
            if (acc_adv1>args.tau and acc_adv2>args.tau and acc_adv3>args.tau and acc_adv4>args.tau): break
        elif (cnt==5):
            print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3,acc_adv4,acc_adv5)
            if (acc_adv1>args.tau and acc_adv2>args.tau and acc_adv3>args.tau and acc_adv4>args.tau and acc_adv5>args.tau): break
        # print(i,loss.item(),loss_a.item(),loss_w.item(),loss_i.item())
        # if (acc_adv1>args.tau): break

    model.encoder.eval()
    model.decoder.eval()
    xr=model.encoder(x0,real_watermark)
    xr = transform_image(xr)
    return xr

def train(model,data,args):
    # WEvade.
    start_time = time.time()
    model.encoder.eval()
    model.decoder.eval()
    criterion = nn.MSELoss()

    # method_list = ['wevade','jpeg', 'gaussian', 'gaussianblur', 'brightness']
    method_list = ['jpeg']
    for method in method_list:
        Bit_acc = AverageMeter()
        Clean_psnr=AverageMeter()
        Perturbation = AverageMeter()
        Evasion_rate = AverageMeter()
        Batch_time = AverageMeter()
        print("finetuning encoder: ")
        
        for i in range (args.epochs):
            for idx, (image, _) in tqdm(enumerate(data), total=len(data)):
                x0 = transform_image(image).to(device)
                random_watermark = torch.Tensor(np.random.choice([0, 1], (image.shape[0], args.watermark_length))).to(device)

                xr=update_model(model,x0,random_watermark,criterion,args,"encoder")
                # xr=direct_optimize(model,x0,random_watermark,criterion,args)
                xr = transform_image(xr)
                xra=apply_attack(x0,xr,random_watermark, model.decoder, 'wevade',criterion, args)

                decoded_watermark = model.decoder(xr)
                rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
                acc_clean = 1 - np.sum(np.abs(rounded_decoded_watermark - random_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)

                decoded_watermark = model.decoder(xra)
                rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
                acc_adv = 1 - np.sum(np.abs(rounded_decoded_watermark - random_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)
                # Detection for double-tailed/single-tailed detector.
                if args.detector_type == 'double-tailed':
                    evasion = (1-args.tau <= acc_adv and acc_adv <= args.tau)
                elif args.detector_type == 'single-tailed':
                    evasion = (acc_adv <= args.tau)

                image=np.asarray(x0.detach().cpu().numpy())
                image_wm=np.asarray(xr.detach().cpu().numpy())
                image_attack=np.asarray(xra.detach().cpu().numpy())
                psnr_clean=cal_psnr(image,image_wm)
                ssim_clean=cal_psnr(image,image_wm)
                lpips_score_clean=cal_lpips(image,image_wm)
                print("cur epochs/idx: ",i,"/",idx)
                print("For watermark, PSNR:",psnr_clean,"SSIM:",ssim_clean,"LPIPS:",lpips_score_clean,"ACC:",acc_clean)
                psnr_adv=cal_psnr(image,image_attack)
                ssim_adv=cal_psnr(image,image_attack)
                lpips_score_adv=cal_lpips(image,image_attack)
                print("For attacks, PSNR:",psnr_adv,"SSIM:",ssim_adv,"LPIPS:",lpips_score_adv,"ACC:",acc_adv)
                # bound = bound / 2   # [-1,1]->[0,1]
                Bit_acc.update(acc_adv, image.shape[0])
                Clean_psnr.update(psnr_clean, image.shape[0])
                Perturbation.update(psnr_adv, image.shape[0])
                Evasion_rate.update(evasion, image.shape[0])
                Batch_time.update(time.time() - start_time)
                start_time = time.time()

                checkpoint = {
                    'enc-model': model.encoder.state_dict(),
                    'dec-model': model.decoder.state_dict()
                }
                if ((idx+1)%10==0): torch.save(checkpoint, "./finetuned/epoch_"+str(idx+1)+"_"+str(args.lr_encoder)+"_"+str(args.lr_decoder)+"_"+str(args.iter_finetune)+"_"+str(args.lr_jpeg)+"_"+str(args.attack_train)+"_"+str(args.info)+".pth")
            print("Average Bit_acc=%.4f\t, Average Clean_psnr=%.4f\t, Average Perturbation=%.4f\t Evasion rate=%.2f\t Time=%.2f" % (Bit_acc.avg,psnr_clean, Perturbation.avg, Evasion_rate.avg, Batch_time.sum))
        
def eval(model,data,args):
    # WEvade.
    start_time = time.time()
    model.encoder.eval()
    model.decoder.eval()
    criterion = nn.MSELoss()

    method_list = ['regen-cheng','regen-bmsh','regen-diff','wevade','combined','jpeg', 'gaussian', 'gaussianblur', 'brightness']
    method_list = ['wevade']
    for method in method_list:
        print("method:", method)
        folder="./result/"+method
        os.makedirs(folder, exist_ok=True)
        Clean_acc=AverageMeter()
        Bit_acc = AverageMeter()
        Clean_psnr=AverageMeter()
        Perturbation = AverageMeter()
        Evasion_rate = AverageMeter()
        Batch_time = AverageMeter()
        
        for idx, (image, _) in tqdm(enumerate(data), total=len(data)):
            x0 = transform_image(image).to(device)
            random_watermark = torch.Tensor(np.random.choice([0, 1], (image.shape[0], args.watermark_length))).to(device)

            xr=model.encoder(x0,random_watermark)
            xr=direct_optimize(model,x0,random_watermark,criterion,args)
            xr = transform_image(xr)
            
            xra=apply_attack(x0,xr,random_watermark, model.decoder, method,criterion, args)

            decoded_watermark = model.decoder(xr)
            rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
            acc_clean = 1 - np.sum(np.abs(rounded_decoded_watermark - random_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)

            decoded_watermark = model.decoder(xra)
            rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
            acc_adv = 1 - np.sum(np.abs(rounded_decoded_watermark - random_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)
            # Detection for double-tailed/single-tailed detector.
            if args.detector_type == 'double-tailed':
                evasion = (1-args.tau <= acc_adv and acc_adv <= args.tau)
            elif args.detector_type == 'single-tailed':
                evasion = (acc_adv <= args.tau)

            image=np.asarray(x0.detach().cpu().numpy())
            image_wm=np.asarray(xr.detach().cpu().numpy())
            image_attack=np.asarray(xra.detach().cpu().numpy())
            
            save_images(x0,xr,xra,folder+"/"+str(idx)+".png")
            psnr_clean=cal_psnr(image,image_wm)
            ssim_clean=cal_ssim(image,image_wm)
            lpips_score_clean=cal_lpips(image,image_wm)
            print("cur idx: ",idx)
            print("For watermark, PSNR:",psnr_clean,"SSIM:",ssim_clean,"LPIPS:",lpips_score_clean,"ACC:",acc_clean)
            psnr_adv=cal_psnr(image,image_attack)
            ssim_adv=cal_ssim(image,image_attack)
            lpips_score_adv=cal_lpips(image,image_attack)
            print("For attacks, PSNR:",psnr_adv,"SSIM:",ssim_adv,"LPIPS:",lpips_score_adv,"ACC:",acc_adv)
            # bound = bound / 2   # [-1,1]->[0,1]
            Clean_acc.update(acc_clean, image.shape[0])
            Bit_acc.update(acc_adv, image.shape[0])
            Clean_psnr.update(psnr_clean, image.shape[0])
            Perturbation.update(psnr_adv, image.shape[0])
            Evasion_rate.update(evasion, image.shape[0])
            Batch_time.update(time.time() - start_time)
            start_time = time.time()
        print("Average Clean_acc=%.4f\t, Average Clean_psnr=%.4f\t, Average Bit_acc=%.4f\t, Average Perturbation=%.4f\t Evasion rate=%.2f\t Time=%.2f" % (Clean_acc.avg,Clean_psnr.avg,Bit_acc.avg, Perturbation.avg, Evasion_rate.avg, Batch_time.sum))

def main():
    parser = argparse.ArgumentParser(description='WEvade-W Arguments.')
    parser.add_argument('--checkpoint', default='./converted/clean.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='../coco/train', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=128, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--decoder_blocks_num', default=7, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--batch', default=10, type=int, help='batch size.')
    parser.add_argument('--tau', default=0.8, type=float, help='Detection threshold of the detector.')
    parser.add_argument('--iteration', default=200, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--delta_finetune', default=80/255, type=float, help='Max perturbation for finetuning.')

    parser.add_argument('--mode', default="eval", type=str, help='eval model.')
    parser.add_argument('--iter_finetune', default=300, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--epochs', default=1, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--lr_encoder', default=1e-3, type=float, help='Max perturbation for finetuning.')
    parser.add_argument('--lr_decoder', default=1e-3, type=float, help='Max perturbation for finetuning.')
    parser.add_argument('--lr_image', default=1e-2, type=float, help='Max perturbation for finetuning.')
    parser.add_argument('--lr_jpeg', default=0, type=int, help='Max perturbation for finetuning.')
    parser.add_argument('--lamda_i', default=10, type=float, help='Max perturbation for finetuning.')
    parser.add_argument('--attack_train', default="wevade", type=str, help='attack.')
    parser.add_argument('--attack_train1', default="jpeg", type=str, help='attack.')
    parser.add_argument('--info', default="onlywevade", type=str, help='attack.')

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
    model = Model(args.image_size, args.watermark_length,args.decoder_blocks_num, device)
    checkpoint = torch.load(args.checkpoint)
    # print(checkpoint['enc-model'])
    model.encoder.load_state_dict(checkpoint['enc-model'])
    model.decoder.load_state_dict(checkpoint['dec-model'])

    # checkpoint = torch.load("./converted/clean.pth")
    # print(checkpoint['enc-model'])
    # model.encoder.load_state_dict(checkpoint['enc-model'])
    # model.decoder.load_state_dict(checkpoint['dec-model'])

    # Load dataset.
    data = get_data_loaders(args.image_size, args.dataset_folder,args.batch)
    if (args.mode=="train"): train(model,data,args)
    elif (args.mode=="eval"): eval(model,data,args)

if __name__ == '__main__':
    main()