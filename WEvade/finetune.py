import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from tqdm import tqdm
from diffusers import ReSDPipeline,StableDiffusionPipeline
from main_WEvade_B_Q import get_watermark_detector,Class_Layer,WEvade_B_Q,JPEG_initailization
from art.estimators.classification import PyTorchClassifier
import lpips

from utils import *
from model.model import Model
from WEvade import WEvade_W, WEvade_W_binary_search_r
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.brightness import Brightness
from noise_layers.wmattacker import *

# np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_fn = lpips.LPIPS(net='vgg')  # 或者 'alex' 也是可选的
loss_fn = loss_fn.to(device)

pipe = ReSDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
# pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe.set_progress_bar_config(disable=True)
pipe.to(device)
print('Finished loading model')

pipe1 = ReSDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# pipe1 = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe1.set_progress_bar_config(disable=True)
pipe1.to(device)
print('Finished loading model')

attackers = {
    'diff_1': DiffWMAttacker(pipe, batch_size=5, noise_step=0, captions={}),
    'diff_2': DiffWMAttacker(pipe1, batch_size=5, noise_step=0, captions={}),
    'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
    'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
    'jpeg_attacker_50': JPEGAttacker(quality=50),
}

def black_attack(x0,xr,real_watermark, Decoder, method,criterion, args):
    watermarked_images=xr.detach().cpu().numpy()
    quality_ls = [99,90,70,50,30,10,5,3,2,1]
    th_ls = [args.tau]
    labels = np.ones((len(watermarked_images)))
    verbose=False
    # th_ls = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ### Attack   
    for th in th_ls:
        # print("############################## THRESHOLD {} ##############################".format(th))
        ### Initialize watermark detector
        detector = get_watermark_detector(Decoder, real_watermark[0], args.detector_type, th, device)
        detector = PyTorchClassifier(
            model=detector,
            clip_values=(-1.0, 1.0),
            input_shape=(3, args.image_size, args.image_size),
            nb_classes=2,
            use_amp=False,
            channels_first=True,
            loss=None,
        )
        
        ### JPEG initialization
        init_adv_images, num_queries_ls = JPEG_initailization(watermarked_images, labels, detector, quality_ls, natural_adv=None, verbose=verbose)

        ### Run WEvade-B-Q
        # print(args, watermarked_images.shape, init_adv_images.shape, detector, num_queries_ls, verbose)
        best_adv_images, saved_num_queries_ls = WEvade_B_Q(args, watermarked_images, init_adv_images, detector, num_queries_ls, verbose=verbose)
        # print("Average number of queries: {}\n".format(np.sum(saved_num_queries_ls)/len(best_adv_images)))

        # print(best_adv_images.shape,best_adv_images)
        best_adv_images=torch.from_numpy(best_adv_images).float().to(device)
        return best_adv_images

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
        name='diff_1'
        noise_layer=attackers[name]
        image = noise_layer.attack(image,eval=eval)
        image=2*image-1
        image=transform_image(image)
    elif method == 'regen-diff-1':
        name='diff_2'
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
    elif method == 'black':
        image=black_attack(x0,xr,real_watermark, Decoder, method,criterion, args)
    else:
        pass

    return image

def pre_optimize(model,x0,real_watermark,criterion,args):
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
    acc_adv1=0
    acc_adv2=0
    acc_adv3=0
    acc_adv4=0
    acc_adv5=0

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

        loss_a=0
        cnt=1e-9

        loss_a1,acc_adv1,psnr_adv1=cal_xra(x0,xr,"wevade");weight1=1;loss_a+=weight1*loss_a1;cnt+=weight1

        # loss_a2,acc_adv2,psnr_adv2=cal_xra(x0,xr,args.attack_train1);weight2=1
        # # if (acc_adv2<=args.tau): loss_a+=weight2*loss_a2;cnt+=weight2

        # loss_a3,acc_adv3,psnr_adv3=cal_xra(x0,xr,"regen-diff");weight3=0.3
        # if (acc_adv3<=args.tau): loss_a+=weight3*loss_a3;cnt+=weight3

        # loss_a4,acc_adv4,psnr_adv4=cal_xra(x0,xr,"gaussian");weight4=0.1
        # if (acc_adv4<=0.9): loss_a+=weight4*loss_a4;cnt+=weight4

        # loss_a5,acc_adv5,psnr_adv5=cal_xra(x0,xr,"gaussianblur");weight5=0.1
        # if (acc_adv5<=0.9): loss_a+=weight5*loss_a5;cnt+=weight5

        # loss_a5,acc_adv5,psnr_adv5=cal_xra(x0,xr,"regen-diff");weight=0.1;loss_a+=weight*loss_a5;cnt+=weight

        loss_a=loss_a/cnt
        loss_w = criterion(decoded_watermark_clean, real_watermark)
        loss_i1=0.1*torch.mean(loss_fn(x0,xr))+0.9*criterion(x0,xr)
        loss_i2=criterion(tmp,xr)
        loss_i=(1*loss_i1+0*loss_i2)/1

        loss=0.10*loss_a+0.1*loss_w+10*loss_i

        grads = torch.autograd.grad(loss, xr)
        lr_tmp=1e-3
        xr = xr - lr_tmp * torch.sign(grads[0])
        delta=xr - x0
        delta=torch.clamp(delta,-2*args.delta_finetune,2*args.delta_finetune)
        # print(i,loss,bound)
        xr=x0+delta
        xr = transform_image(xr)
        xr=xr.detach()
        print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3,acc_adv4,acc_adv5)
        if (psnr_clean>=33.5): break
    model.encoder.eval()
    model.decoder.eval()
    return xr

def direct_optimize(model,x0,real_watermark,criterion,args,xr_pre):
    model.encoder.eval()
    model.decoder.eval()
    
    # tmp=model.encoder(x0,real_watermark)
    # xr_pre=transform_image(tmp)
    xr=xr_pre.clone().detach().requires_grad_(True)
    params = [
        # {'params': model.encoder.parameters(), 'lr': args.lr_encoder},
        {'params': model.decoder.parameters(), 'lr': args.lr_decoder}  # 设置decoder参数的学习率为 1e-4
    ]

    # 创建优化器时使用 params 参数组设置不同的学习率
    optimizer = torch.optim.Adam(params)
    acc_adv1=0
    acc_adv2=0
    acc_adv3=0
    acc_adv4=0
    acc_adv5=0

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

        loss_a=0
        cnt=1e-9

        loss_a1,acc_adv1,psnr_adv1=cal_xra(x0,xr,args.attack_train);weight1=1
        if (acc_adv1<=0.85): loss_a+=weight1*loss_a1;cnt+=weight1

        loss_a2,acc_adv2,psnr_adv2=cal_xra(x0,xr,args.attack_train1);weight2=1
        # if (acc_adv2<=0.72): loss_a+=weight2*loss_a2;cnt+=weight2

        loss_a3,acc_adv3,psnr_adv3=cal_xra(x0,xr,"regen-diff");weight3=0.1
        if (acc_adv3<=args.tau): loss_a+=weight3*loss_a3;cnt+=weight3

        loss_a4,acc_adv4,psnr_adv4=cal_xra(x0,xr,"gaussian");weight4=0.1
        if (acc_adv4<=0.99): loss_a+=weight4*loss_a4;cnt+=weight4

        loss_a5,acc_adv5,psnr_adv5=cal_xra(x0,xr,"gaussianblur");weight5=0.1
        if (acc_adv5<=0.99): loss_a+=weight5*loss_a5;cnt+=weight5

        # loss_a5,acc_adv5,psnr_adv5=cal_xra(x0,xr,"regen-diff");weight=0.1;loss_a+=weight*loss_a5;cnt+=weight

        loss_a=loss_a/cnt
        loss_w = criterion(decoded_watermark_clean, real_watermark)
        loss_i1=0.1*torch.mean(loss_fn(x0,xr))+0.9*criterion(x0,xr)
        loss_i2=criterion(xr_pre,xr)
        loss_i=(1*loss_i1+0*loss_i2)/2

        loss=loss_a+0.1*loss_w+args.lamda_i*loss_i

        grads = torch.autograd.grad(loss, xr)
        xr = xr - args.lr_image * torch.sign(grads[0])
        delta=xr - x0
        delta=torch.clamp(delta,-2*args.delta_finetune,2*args.delta_finetune)
        # print(i,loss,bound)
        xr=x0+delta
        xr = transform_image(xr)
        xr=xr.detach()
        if (i%10==0): print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3,acc_adv4,acc_adv5)
        if (acc_adv1>0.8 and acc_adv3>0.75): break
        if (psnr_clean<=29.5): break
    model.encoder.eval()
    model.decoder.eval()
    return xr

def update_model(model,x0,real_watermark,criterion,args,select):
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
    acc_adv1=0
    acc_adv2=0
    acc_adv3=0
    acc_adv4=0
    acc_adv5=0
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

        # model.encoder.train()
        # model.decoder.eval()
        xr=model.encoder(x0,real_watermark)
        xr = transform_image(xr)

        decoded_watermark_clean = model.decoder(xr)
        decoded_watermark_clean=torch.clamp(decoded_watermark_clean,0,1)
        acc_clean = 1 - np.sum(np.abs(decoded_watermark_clean.detach().cpu().numpy().round() - real_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)
        psnr_clean=cal_psnr(x0.detach().cpu().numpy(),xr.detach().cpu().numpy())

        loss_a=0
        cnt=1e-9

        loss_a1,acc_adv1,psnr_adv1=cal_xra(x0,xr,args.attack_train);weight1=1;loss_a+=weight1*loss_a1;cnt+=weight1

        # loss_a2,acc_adv2,psnr_adv2=cal_xra(x0,xr,args.attack_train1);weight2=1
        # if (acc_adv2<=args.tau): loss_a+=weight2*loss_a2;cnt+=weight2

        # loss_a3,acc_adv3,psnr_adv3=cal_xra(x0,xr,"regen-diff");weight3=0.3
        # if (acc_adv3<=args.tau): loss_a+=weight3*loss_a3;cnt+=weight3

        # loss_a4,acc_adv4,psnr_adv4=cal_xra(x0,xr,"gaussian");weight4=0.1
        # if (acc_adv4<=0.9): loss_a+=weight4*loss_a4;cnt+=weight4

        # loss_a5,acc_adv5,psnr_adv5=cal_xra(x0,xr,"gaussianblur");weight5=0.1
        # if (acc_adv5<=0.9): loss_a+=weight5*loss_a5;cnt+=weight5

        # loss_a5,acc_adv5,psnr_adv5=cal_xra(x0,xr,"regen-diff");weight=0.1;loss_a+=weight*loss_a5;cnt+=weight

        loss_a=loss_a/cnt
        
        loss_w = criterion(decoded_watermark_clean, real_watermark)
        # loss_i=criterion(x0,xr)
        loss_i=0.5*torch.mean(loss_fn(x0,xr))+0.5*criterion(x0,xr)
        loss=loss_a+1*loss_w+args.lamda_i*loss_i
        
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
        loss.backward()

        if (i==args.iter_finetune-1 and acc_adv1<=args.tau): optimizer_de.step()
        else: optimizer_en.step()

        # optimizer_en.step()
        # optimizer_de.step()
        print(i,loss.item(),acc_clean,psnr_clean,acc_adv1,acc_adv2,acc_adv3,acc_adv4,acc_adv5)
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

    method_list = ['wevade','jpeg', 'gaussian', 'gaussianblur', 'brightness']
    method_list = ['wevade']
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
                psnr_clean=cal_psnr(image,image_wm)
                ssim_clean=cal_ssim(image,image_wm)
                lpips_score_clean=cal_lpips(image,image_wm)
                print("cur epochs/idx: ",i,"/",idx)
                print("For watermark, PSNR:",psnr_clean,"SSIM:",ssim_clean,"LPIPS:",lpips_score_clean,"ACC:",acc_clean)
                psnr_adv=cal_psnr(image,image_attack)
                ssim_adv=cal_ssim(image,image_attack)
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

    Clean_acc=AverageMeter()
    Clean_psnr=AverageMeter()
    Clean_ssim=AverageMeter()
    Clean_lpips=AverageMeter()

    # Bit_acc = AverageMeter()
    # Perturbation = AverageMeter()
    # Evasion_rate = AverageMeter()
    # Batch_time = AverageMeter()

    method_list = ['wevade','regen-diff','regen-diff-1','combined','jpeg', 'gaussian', 'gaussianblur', 'brightness','black']
    # method_list = ['black']
    Bit_acc={}
    Perturbation={}
    Evasion_rate={}
    Batch_time={}

    for method in method_list:
        Bit_acc[method]=AverageMeter()
        Perturbation[method]=AverageMeter()
        Evasion_rate[method]=AverageMeter()
        Batch_time[method]=AverageMeter()

    opt_xr=[]
    dataset=args.dataset_folder.split("/")[-1]
    xr_pre_all=np.load("./pre_opt/opt_xr_"+str(dataset)+"_"+str(args.lamda_i)+".npy")
    # xr_pre_all=np.load("./result/opt_xr_pre.npy")
    xr_pre_all=torch.from_numpy(xr_pre_all).float().to(device)
    tmp=torch.rand((100,3,128,128)).float().to(device)
    for i in range (xr_pre_all.shape[0]):
        tmp[xr_pre_all.shape[1]*i:xr_pre_all.shape[1]*(i+1),:,:,:]=xr_pre_all[i]
    print(xr_pre_all.shape,tmp.shape)
    xr_pre_all=tmp

    for idx, (image, _) in tqdm(enumerate(data), total=len(data)):
        # if (idx==1): break
        x0 = transform_image(image).to(device)
        random_watermark = torch.Tensor(np.random.choice([0, 1], (image.shape[0], args.watermark_length))).to(device)
        random_watermark=torch.from_numpy(np.load('./watermark/watermark_coco.npy')).float().to(device).repeat(x0.shape[0], 1)
        xr=model.encoder(x0,random_watermark)

        results={}
        xr_pre=xr_pre_all[args.batch*idx:args.batch*(idx+1)]
        # xr_pre=pre_optimize(model,x0,random_watermark,criterion,args)
        # opt_xr.append(xr_pre)
        # continue
        cnt=0
        while (1):
            cnt+=1
            # xr=direct_optimize(model,x0,random_watermark,criterion,args,xr_pre)
            if (args.defense=="advmark"): xr=xr_pre

            xr = transform_image(xr)
            decoded_watermark = model.decoder(xr)
            rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
            acc_clean = 1 - np.sum(np.abs(rounded_decoded_watermark - random_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)

            image=np.asarray(x0.detach().cpu().numpy())
            image_wm=np.asarray(xr.detach().cpu().numpy())
            psnr_clean=cal_psnr(image,image_wm)
            ssim_clean=cal_ssim(image,image_wm)
            lpips_score_clean=cal_lpips(image,image_wm)
            print("cur idx: ",idx)
            print("For watermark, PSNR:",psnr_clean,"SSIM:",ssim_clean,"LPIPS:",lpips_score_clean,"ACC:",acc_clean)
            results['xr']=xr
            results['acc_clean']=acc_clean
            results['psnr_clean']=psnr_clean
            results['ssim_clean']=ssim_clean
            results['lpips_score_clean']=lpips_score_clean

            for method in method_list:
                folder="./result/"+str(dataset)+"/"+args.defense+"/"+method
                os.makedirs(folder, exist_ok=True)
                xra=apply_attack(x0,xr,random_watermark, model.decoder, method,criterion, args)
                decoded_watermark = model.decoder(xra)
                rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
                acc_adv = 1 - np.sum(np.abs(rounded_decoded_watermark - random_watermark.cpu().numpy())) / (xr.shape[0] * args.watermark_length)
                # Detection for double-tailed/single-tailed detector.
                if args.detector_type == 'double-tailed':
                    evasion = (1-args.tau <= acc_adv and acc_adv <= args.tau)
                elif args.detector_type == 'single-tailed':
                    evasion = (acc_adv <= args.tau)

                image_attack=np.asarray(xra.detach().cpu().numpy())
                save_images(x0,xr,xra,folder+"/"+str(idx)+".png")
                psnr_adv=cal_psnr(image,image_attack)
                ssim_adv=cal_ssim(image,image_attack)
                lpips_score_adv=cal_lpips(image,image_attack)
                print("For attacks: ",method," PSNR:",psnr_adv,"SSIM:",ssim_adv,"LPIPS:",lpips_score_adv,"ACC:",acc_adv)
                Bit_acc[method].update(acc_adv, image.shape[0])
                Perturbation[method].update(psnr_adv, image.shape[0])
                Evasion_rate[method].update(evasion, image.shape[0])
                Batch_time[method].update(time.time() - start_time)
                start_time = time.time()
                results[method]=acc_adv
            if (results['psnr_clean']>=29.5 and results['jpeg']>=0.8 and results['regen-diff']>=0.75): break
            if (cnt>=1): break
        opt_xr.append(results['xr'])
        Clean_acc.update(results['acc_clean'], image.shape[0])
        Clean_psnr.update(results['psnr_clean'], image.shape[0])
        Clean_ssim.update(results['ssim_clean'], image.shape[0])
        Clean_lpips.update(results['lpips_score_clean'], image.shape[0])

        for method in method_list:
            Bit_acc[method].update(results[method], image.shape[0])
        
    print("Average Clean_acc=%.4f\t, Average Clean_psnr=%.4f\t, Average Clean_ssim=%.4f\t, Average Clean_lpips=%.4f\t" % (Clean_acc.avg,Clean_psnr.avg,Clean_ssim.avg,Clean_lpips.avg))
    for method in method_list:
        print("Attack: ",method)
        print("Average Bit_acc=%.4f\t, Average Perturbation=%.4f\t Evasion rate=%.2f\t Time=%.2f" % (Bit_acc[method].avg, Perturbation[method].avg, Evasion_rate[method].avg, Batch_time[method].sum))

    # x_stacked = torch.stack(opt_xr, dim=0).detach().cpu().numpy()
    # # np.save("./result/opt_xr_"+str(args.lamda_i)+".npy", x_stacked)
    # # np.save("./result/opt_xr_pre.npy", x_stacked)
    # # np.save("./result/opt_xr_pre_db.npy", x_stacked)
    # np.save("./result/opt_xr_db_"+str(args.lamda_i)+".npy", x_stacked)

def main():
    parser = argparse.ArgumentParser(description='WEvade-W Arguments.')
    parser.add_argument('--checkpoint', default='./converted/clean.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='../coco/train', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=128, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--decoder_blocks_num', default=7, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--batch', default=10, type=int, help='batch size.')
    parser.add_argument('--tau', default=0.8, type=float, help='Detection threshold of the detector.')
    parser.add_argument('--iteration', default=100, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--delta_finetune', default=80/255, type=float, help='Max perturbation for finetuning.')

    #Black Attack settings
    parser.add_argument('--batch-size', default=256, type=int, help='batch size for hopskipjump')
    parser.add_argument('--num-attack', default=100, type=int, help='number of images to attack')
    parser.add_argument('--budget', default=2000, type=int, help='query budget')
    parser.add_argument('--init-eval', default=5, type=int, help='hopskipjump parameters')
    parser.add_argument('--max-eval', default=1000, type=int, help='hopskipjump parameters')
    parser.add_argument('--iter-step', default=1, type=int, help='print interval')
    parser.add_argument('--ES', default=20, type=int, help='early stopping criterion')
    parser.add_argument('--norm', default='inf', choices=['2','inf'], help='norm metric') # We optimize different norm for Hopskipjump when using different norm as the metric following their original work

    parser.add_argument('--mode', default="eval", type=str, help='eval model.')
    parser.add_argument('--defense', default="advmark", type=str, help='eval model.')
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