import sys
import torch
import os
import glob
import numpy as np

from diffusers import ReSDPipeline

from utils import eval_psnr_ssim_msssim, bytearray_to_bits
from watermarker import InvisibleWatermarker
from wmattacker import DiffWMAttacker, VAEWMAttacker, JPEGAttacker

wm_text = 'test'
device = 'cuda:0'
ori_path = 'examples/ori_imgs/'
output_path = 'examples/wm_imgs/'
print_width = 50

os.makedirs(output_path, exist_ok=True)
ori_img_paths = glob.glob(os.path.join(ori_path, '*.*'))
ori_img_paths = sorted([path for path in ori_img_paths if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
print(ori_img_paths)

wmarkers = {
    'DwtDct': InvisibleWatermarker(wm_text, 'dwtDct'),
    'DwtDctSvd': InvisibleWatermarker(wm_text, 'dwtDctSvd'),
    'RivaGAN': InvisibleWatermarker(wm_text, 'rivaGan'),
}

# print(ReSDPipeline.save_pretrained)

pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
pipe.set_progress_bar_config(disable=True)
pipe.to(device)
print('Finished loading model')

attackers = {
    'diff_attacker_60': DiffWMAttacker(pipe, batch_size=5, noise_step=60, captions={}),
    'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
    'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
    'jpeg_attacker_50': JPEGAttacker(quality=50),
}

def add_watermark(wmarker_name, wmarker):
    print('*' * print_width)
    print(f'Watermarking with {wmarker_name}')
    os.makedirs(os.path.join(output_path, wmarker_name + '/noatt'), exist_ok=True)
    for ori_img_path in ori_img_paths:
        img_name = os.path.basename(ori_img_path)
        wmarker.encode(ori_img_path, os.path.join(output_path, wmarker_name + '/noatt', img_name))

for wmarker_name, wmarker in wmarkers.items():
    add_watermark(wmarker_name, wmarker)
print('Finished watermarking')

for wmarker_name, wmarker in wmarkers.items():
    for attacker_name, attacker in attackers.items():
        print('*' * print_width)
        print(f'Attacking {wmarker_name} with {attacker_name}')
        wm_img_paths = []
        att_img_paths = []
        os.makedirs(os.path.join(output_path, wmarker_name, attacker_name), exist_ok=True)
        for ori_img_path in ori_img_paths:
            img_name = os.path.basename(ori_img_path)
            wm_img_paths.append(os.path.join(output_path, wmarker_name + '/noatt', img_name))
            att_img_paths.append(os.path.join(output_path, wmarker_name, attacker_name, img_name))
        attackers[attacker_name].attack(wm_img_paths, att_img_paths)

print('Finished attacking')

wm_results = {}
for wmarker_name, wmarker in wmarkers.items():
    print('*' * print_width)
    print(f'Watermark: {wmarker_name}')
    wm_successes = []
    wm_psnr_list = []
    wm_ssim_list = []
    wm_msssim_list = []
    for ori_img_path in ori_img_paths:
        img_name = os.path.basename(ori_img_path)
        wm_img_path = os.path.join(output_path, wmarker_name+'/noatt', img_name)
        wm_psnr, wm_ssim, wm_msssim = eval_psnr_ssim_msssim(ori_img_path, wm_img_path)
        print(wm_psnr, wm_ssim, wm_msssim)
        wm_psnr_list.append(wm_psnr)
        wm_ssim_list.append(wm_ssim)
        wm_msssim_list.append(wm_msssim)
    wm_results[wmarker_name] = {}
    wm_results[wmarker_name]['wm_psnr'] = np.array(wm_psnr_list).mean()
    wm_results[wmarker_name]['wm_ssim'] = np.array(wm_ssim_list).mean()
    wm_results[wmarker_name]['wm_msssim'] = np.array(wm_msssim_list).mean()

print('Finished evaluating watermarking')

detect_wm_results = {}
for wmarker_name, wmarker in wmarkers.items():
    print('*' * print_width)
    print(f'Watermark: {wmarker_name}')
    bit_accs = []
    wm_successes = []
    for ori_img_path in ori_img_paths:
        img_name = os.path.basename(ori_img_path)
        wm_img_path = os.path.join(output_path, wmarker_name+'/noatt', img_name)
        wm_text = wmarkers[wmarker_name].decode(wm_img_path)
        try:
            if type(wm_text) == bytes:
                a = bytearray_to_bits('test'.encode('utf-8'))
                b = bytearray_to_bits(wm_text)
            elif type(wm_text) == str:
                a = bytearray_to_bits('test'.encode('utf-8'))
                b = bytearray_to_bits(wm_text.encode('utf-8'))
            bit_acc = (np.array(a) ==  np.array(b)).mean()
            bit_accs.append(bit_acc)
            if bit_acc > 24/32:
                wm_successes.append(img_name)
        except:
            print('#' * print_width)
            print(f'failed to decode {wm_text}', type(wm_text), len(wm_text))
            pass
    detect_wm_results[wmarker_name] = {}
    detect_wm_results[wmarker_name]['bit_acc'] = np.array(bit_accs).mean()
    detect_wm_results[wmarker_name]['wm_success'] = len(wm_successes) / len(ori_img_paths)
print('Finished evaluating watermarking')

detect_att_results = {}
for wmarker_name, wmarker in wmarkers.items():
    print('*' * print_width)
    print(f'Watermark: {wmarker_name}')
    detect_att_results[wmarker_name] = {}
    for attacker_name, attacker in attackers.items():
        print(f'Attacker: {attacker_name}')
        bit_accs = []
        wm_successes = []
        for ori_img_path in ori_img_paths:
            img_name = os.path.basename(ori_img_path)
            att_img_path = os.path.join(output_path, wmarker_name, attacker_name, img_name)
            att_text = wmarkers[wmarker_name].decode(att_img_path)
            try:
                if type(att_text) == bytes:
                    a = bytearray_to_bits('test'.encode('utf-8'))
                    b = bytearray_to_bits(att_text)
                elif type(att_text) == str:
                    a = bytearray_to_bits('test'.encode('utf-8'))
                    b = bytearray_to_bits(att_text.encode('utf-8'))
                bit_acc = (np.array(a) ==  np.array(b)).mean()
                bit_accs.append(bit_acc)
                if bit_acc > 24/32:
                    wm_successes.append(img_name)
            except:
                print('#' * print_width)
                print(f'failed to decode {wm_text}', type(wm_text), len(wm_text))
                pass
        detect_att_results[wmarker_name][attacker_name] = {}
        detect_att_results[wmarker_name][attacker_name]['bit_acc'] = np.array(bit_accs).mean()
        detect_att_results[wmarker_name][attacker_name]['wm_success'] = len(wm_successes) / len(ori_img_paths)


from IPython.display import Image
img_id = '000000000711.png'
Image(filename='examples/ori_imgs/'+img_id) # original image

Image(filename='examples/wm_imgs/DwtDct/noatt/'+img_id) # watermarked image

Image(filename='examples/wm_imgs/DwtDct/diff_attacker_60/'+img_id) # diffusion attacker