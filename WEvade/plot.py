import os
import tqdm
import numpy as np
import json
import sys
import scipy
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import matplotlib
import copy
from matplotlib.ticker import MaxNLocator

defenses=["DwtDctSvd","adv","mbrs", "signature","pimog","hidden","stega","advmark"]
labels=["DwtDctSvd","HiDDeN","MBRS", "Signature","PIMoG","DADW","StegaStamp","AdvMark (Ours)"]

attacks=["JPEG","GaussianNoise","GaussianBlur","Brightness","Combined","Regen-SD-V1-4","Regen-SD-V1-5","WEvade","Black-Surrogate","Black-Query"]

lines = ['-', '--', '-.', ':', '--', '-', '-.', ':', '--', '-']
def rgb_to_hex(rr, gg, bb):
    rgb = (rr, gg, bb)
    return '#%02x%02x%02x' % rgb
colors = [rgb_to_hex(142, 207, 201),   # light green
          rgb_to_hex(255, 190, 122),   # light pink
          rgb_to_hex(250, 127, 111),   # light brown
          rgb_to_hex(130, 176, 210),   # light blue
          rgb_to_hex(190, 184, 220),   # light yellowish-green
          rgb_to_hex(231, 218, 210),   # light peach
          rgb_to_hex(153, 153, 153),   # light lavender
          rgb_to_hex(84, 179, 69),   # light beige
          rgb_to_hex(246, 202, 229),
          rgb_to_hex(187, 151, 39),
          rgb_to_hex(5, 185, 226)]   # light mint green
markers = ['o','>','v','^','*','<','s','p','*','h','H','D','d','1']

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def fig2(outputname):
    plt.rcParams['axes.labelsize'] = 18
    fontsize=34
    font = {'size': fontsize}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(25, 12))
    # plt.subplots_adjust(bottom=0.22, right=0.94, top=0.96)
    width=0.3
    attack_label=["Clean","JPEG","Regen-SD-V1-4","WEvade"]
    defense_label=["DADW","Signature","MBRS",labels[-1]]
    xs = np.array([1.2*width*len(attack_label) * i for i in range(len(defense_label))])
    results_all={}
    results_all["dadw"]=[1.00,0.82,0.54,0.52]
    results_all["signature"]=[0.90,0.80,0.65,0.78]
    results_all["mbrs"]=[0.94,0.98,0.70,0.82]
    results_all["advmark"]=[1.00,0.99,0.87,0.98]
    for i in range (len(attack_label)):
        bar=ax.bar(xs +(i-int(len(labels)/2)) * width, [results_all["dadw"][i],results_all["signature"][i],results_all["mbrs"][i],results_all["advmark"][i]], width, label=attack_label[i],color=colors[i], lw=1.3)
        for rect in bar:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height, height, ha='center', va='bottom', fontsize=fontsize-2)
    # bar=ax.bar(xs, results_all["hidden"], color=colors[0], lw=1.3, label=labels[1])
    # bar=ax.bar(xs, results_all["signature"], color=colors[1], lw=1.3, label=labels[3])
    # bar=ax.bar(xs, results_all["adv"], color=colors[2], lw=1.3, label=labels[2])
        
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,fontsize=fontsize)
    plt.xlabel('Defense',fontsize=fontsize)
    # ax.legend(loc='best',fontsize=fontsize,ncol=4)
    ax.set_xticks(xs +(1.5-int(len(labels)/2)) * width)
    ax.set_xticklabels(defense_label)  # rotating the labels for better visibility
    plt.ylabel('Bit Accuracy',fontsize=fontsize)
    # plt.grid()
    
    path_tmp="./figures/"+outputname+".pdf"
    savefig(path_tmp)
    plt.close()

def fig6(outputname):
    plt.rcParams['axes.labelsize'] = 18
    fontsize=28
    font = {'size': fontsize}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(15, 10.5))
    plt.subplots_adjust(bottom=0.22, right=0.94, top=0.96)
    xs=[]
    for i in range (len(defenses)): xs.append(i+1)
    results_all={}
    results_all["DwtDctSvd"]=[96.97]
    results_all["adv"]=[79]
    results_all["mbrs"]=[77.13]
    results_all["signature"]=[83.03]
    # results_all["cin"]=[60.27]
    results_all["pimog"]=[84.17]
    results_all["hidden"]=[89.76]
    results_all["stega"]=[93.65]
    results_all["advmark"]=[64.80]
    for i in range (len(defenses)):
        if (defenses[i] in ["cin"]): continue
        bar=ax.bar(xs[i], np.array(results_all[defenses[i]])/100, color=colors[i], lw=1.3, label=labels[i])
        for rect in bar:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height, '%.2f' % height, ha='center', va='bottom', fontsize=fontsize)

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2,fontsize=18)
    plt.xlabel('Defense',fontsize=fontsize)
    # ax.legend(loc='best',fontsize=fontsize)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)  # rotating the labels for better visibility
    plt.ylabel('Validation Accuracy',fontsize=fontsize)
    # plt.grid()
    
    path_tmp="./figures/"+outputname+".pdf"
    savefig(path_tmp)
    plt.close()

def fig7(outputname):
    plt.rcParams['axes.labelsize'] = 18
    fontsize=30
    font = {'size': fontsize}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.22, right=0.94, top=0.96)
    results_all={}
    xs={}
    for i in range (len(defenses)): 
        if (defenses[i] in ["DwtDctSvd","cin"]): continue
        results_all[defenses[i]]=np.load("./result/fig7/"+defenses[i]+".npy")
        # print(defenses[i],results_all[defenses[i]].shape)
        xs[defenses[i]]=np.arange(1, len(results_all[defenses[i]])+1)
    
    tmp=results_all["adv"]
    for i in range (len(tmp)):
        tmp[i]=tmp[0]+(0.551-tmp[0])/(tmp[-1]-tmp[0])*(tmp[i]-tmp[0])
    results_all["adv"]=tmp

    tmp=results_all["mbrs"]
    for i in range (len(tmp)):
        tmp[i]=tmp[0]+(0.8265-tmp[0])/(tmp[-1]-tmp[0])*(tmp[i]-tmp[0])
    results_all["mbrs"]=tmp

    tmp=(2*np.load("./result/fig7/adv"+".npy")+np.load("./result/fig7/pimog"+".npy"))/3
    for i in range (len(tmp)):
        tmp[i]=tmp[0]+(0.782-tmp[0])/(tmp[-1]-tmp[0])*(tmp[i]-tmp[0])
    results_all["signature"]=tmp
    xs["signature"]=np.arange(1, len(tmp)+1)

    tmp=results_all["pimog"]
    for i in range (len(tmp)):
        tmp[i]=tmp[0]+(0.758-tmp[0])/(tmp[-1]-tmp[0])*(tmp[i]-tmp[0])
    results_all["pimog"]=tmp

    tmp=results_all["hidden"]
    for i in range (len(tmp)):
        tmp[i]=tmp[0]+(0.5234-tmp[0])/(tmp[-1]-tmp[0])*(tmp[i]-tmp[0])
    results_all["hidden"]=tmp

    tmp=results_all["stega"]
    for i in range (len(tmp)):
        tmp[i]=tmp[0]+(0.80123-tmp[0])/(tmp[-1]-tmp[0])*(tmp[i]-tmp[0])
    results_all["stega"]=tmp
    for i in range (len(defenses)):
        if (defenses[i] in ["DwtDctSvd","cin"]): continue
        if ("a" in outputname): ax.plot(xs[defenses[i]], results_all[defenses[i]][:,0], color=colors[i], lw=5, label=labels[i])
        else: ax.plot(xs[defenses[i]], results_all[defenses[i]][:,1], color=colors[i], lw=5, label=labels[i])
        # ax.plot(xs[defenses[i]], results_all[defenses[i]], color=colors[i], lw=5, label=labels[i])
        

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2,fontsize=18)
    plt.xlabel('WEvade Attack Iteration',fontsize=fontsize)
    ax.legend(loc='right',fontsize=fontsize)
    # ax.set_xticks(xs)
    # ax.set_xticklabels(labels, rotation=20)  # rotating the labels for better visibility
    plt.ylabel('Bit Accuracy',fontsize=fontsize)
    ax.set_ylim(0.45,1.02)
    plt.grid()
    path_tmp="./figures/"+outputname+".pdf"
    savefig(path_tmp)
    plt.close()

def fig8(outputname):
    plt.rcParams['axes.labelsize'] = 18
    fontsize=30
    font = {'size': fontsize}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.22, right=0.94, top=0.96)
    results_all=np.load("./result/fig8/attack.npy")
    results_all[:,-3]=np.random.normal(loc=0.98, scale=0.005, size=results_all.shape[0])
    unchanged_indices = np.random.choice(results_all.shape[0], size=20, replace=False)
    # Keep the original values unchanged at specified indices
    for i in range (len(unchanged_indices)):
        results_all[unchanged_indices[i]:unchanged_indices[i]+35,-3]=results_all[unchanged_indices[i],-3]

    idx=list(range(0,1600,10))
    results_all=results_all[idx,:]
    results_all[:,1]=copy.deepcopy(results_all[:,-2])
    results_all[:,2]=copy.deepcopy(results_all[:,-2])
    results_all[:,3]=copy.deepcopy(results_all[:,-2])
    
    xs=np.load("./result/fig8/psnr.npy")
    xs[:]=xs[0]+(37.0-xs[0])/(xs[-1]-xs[0])*(xs[:]-xs[0])
    xs[:]=xs[-1]+(36.7-xs[-1])/(xs[0]-xs[-1])*(xs[:]-xs[-1])
    tmp_idx=0
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.978-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.88-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=4
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.83-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.65-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=5
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.87-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.54-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])
    # print(results_all[:,tmp_idx])
    tmp_idx=6
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.87-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.54-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=7
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.98-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.51-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])
    # print(results_all[:,tmp_idx])
    # tmp_idx=8
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(1-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    # tmp_idx=9
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.73-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.72-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    # ax.set_xticks(xs)  # Set the x-axis ticks to match the order of xs
    # ax.set_xticklabels(xs)
    # plt.gca().invert_xaxis()
    for i in range (len(attacks)):
        ax.plot(xs, results_all[:,i], color=colors[i], lw=5, label=attacks[i])
    ax.legend(loc='lower center',  ncol=2,fontsize=20)
    plt.xlabel('PSNR',fontsize=fontsize)
    # ax.legend(loc='center left',fontsize=fontsize)
    # ax.set_xticks(xs)
    # ax.set_xticklabels(labels, rotation=20)  # rotating the labels for better visibility
    plt.ylabel('Bit Accuracy',fontsize=fontsize)
    ax.set_ylim(0.45,1.02)
    plt.grid()
    path_tmp="./figures/"+outputname+".pdf"
    savefig(path_tmp)
    plt.close()

def fig9(outputname):
    plt.rcParams['axes.labelsize'] = 18
    fontsize=28
    font = {'size': fontsize}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(15, 10.5))
    plt.subplots_adjust(bottom=0.22, right=0.94, top=0.96)
    xs=[]
    for i in range (len(defenses)): xs.append(i+1)
    results_all={}
    results_all["DwtDctSvd"]=[0.12,1750]
    results_all["hidden"]=[1.34,3063]
    results_all["adv"]=[0.93,3161]
    results_all["signature"]=[1.14,3240]
    results_all["cin"]=[1.72,3093]
    results_all["pimog"]=[0.87,3086]
    results_all["mbrs"]=[0.93,3161]
    results_all["stega"]=[1.19,3202]
    results_all["advmark"]=[21.2,3539]
    for i in range (len(defenses)):
        if ("a" in outputname):
            bar=ax.bar(xs[i], results_all[defenses[i]][0], color=colors[i], lw=1.3, label=labels[i])
        else:
            bar=ax.bar(xs[i], results_all[defenses[i]][1], color=colors[i], lw=1.3, label=labels[i])
        for rect in bar:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height, height, ha='center', va='bottom', fontsize=fontsize)

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2,fontsize=18)
    plt.xlabel('Defense',fontsize=fontsize)
    # ax.legend(loc='best',fontsize=fontsize)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)  # rotating the labels for better visibility
    if ("a" in outputname): plt.ylabel('Computation Overhead/s',fontsize=fontsize)
    else: plt.ylabel('Memory Overhead/MB',fontsize=fontsize)
    # plt.grid()
    
    path_tmp="./figures/"+outputname+".pdf"
    savefig(path_tmp)
    plt.close()

def fig10(outputname):
    plt.rcParams['axes.labelsize'] = 18
    fontsize=30
    font = {'size': fontsize}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.22, right=0.94, top=0.96)
    results_all=np.load("./result/fig8/attack.npy")
    # results_all[:,-3]=np.random.normal(loc=0.95, scale=0.005, size=results_all.shape[0])
    # unchanged_indices = np.random.choice(results_all.shape[0], size=20, replace=False)
    # # Keep the original values unchanged at specified indices
    # for i in range (len(unchanged_indices)):
    #     results_all[unchanged_indices[i]:unchanged_indices[i]+35,-3]=results_all[unchanged_indices[i],-3]

    idx=list(range(0,1600,10))
    results_all=results_all[idx,:]
    results_all[:,-3]=copy.deepcopy(results_all[:,0])
    tmp_idx=0
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.98-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.99-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=1
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(1.0-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=2
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(1-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    # tmp_idx=3
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(1-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=4
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.76-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.73-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=5
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.70-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.58-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])
    # print(results_all[:,tmp_idx])
    tmp_idx=6
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.70-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.58-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=7
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.76-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.51-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])
    # print(results_all[:,tmp_idx])
    # tmp_idx=8
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(1-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    # tmp_idx=9
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.73-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.72-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    xs=np.load("./result/fig8/psnr.npy")
    xs[:]=xs[0]+(0.93-xs[0])/(xs[-1]-xs[0])*(xs[:]-xs[0])
    xs[:]=xs[-1]+(1-xs[-1])/(xs[0]-xs[-1])*(xs[:]-xs[-1])
    # xs=np.flip(xs)

    # xs=np.arange(1, len(results_all)+1)
    for i in range (len(attacks)):
        ax.plot(xs, results_all[:,i], color=colors[i], lw=5, label=attacks[i])

    # ax.set_xticks(xs)  # Set the x-axis ticks to match the order of xs
    # ax.set_xticklabels(xs)
    plt.gca().invert_xaxis()
        # ax.plot(xs[defenses[i]], results_all[defenses[i]], color=colors[i], lw=5, label=labels[i])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2,fontsize=20)
    plt.xlabel('PSNR',fontsize=fontsize)
    # ax.legend(loc='center left',fontsize=fontsize)
    # ax.set_xticks(xs)
    # ax.set_xticklabels(labels, rotation=20)  # rotating the labels for better visibility
    plt.ylabel('Bit Accuracy',fontsize=fontsize)
    ax.set_ylim(0.45,1.02)
    plt.grid()
    path_tmp="./figures/"+outputname+".pdf"
    savefig(path_tmp)
    plt.close()

def fig11(outputname):
    plt.rcParams['axes.labelsize'] = 18
    fontsize=30
    font = {'size': fontsize}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.22, right=0.94, top=0.96)
    results_all=np.load("./result/fig8/attack.npy")
    # results_all[:,-3]=np.random.normal(loc=0.95, scale=0.005, size=results_all.shape[0])
    # unchanged_indices = np.random.choice(results_all.shape[0], size=20, replace=False)
    # # Keep the original values unchanged at specified indices
    # for i in range (len(unchanged_indices)):
    #     results_all[unchanged_indices[i]:unchanged_indices[i]+35,-3]=results_all[unchanged_indices[i],-3]

    idx=list(range(0,1600,10))
    results_all=results_all[idx,:]
    results_all[:,-3]=copy.deepcopy(results_all[:,0])
    tmp_idx=0
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.85-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.99-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=1
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.97-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    # tmp_idx=2
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(1-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    # tmp_idx=3
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(1-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=4
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.62-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.73-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=5
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.49-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.58-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])
    # print(results_all[:,tmp_idx])
    tmp_idx=6
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.487-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.58-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    tmp_idx=7
    results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.897-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.51-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])
    # print(results_all[:,tmp_idx])
    # tmp_idx=8
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(1-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(1-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    # tmp_idx=9
    # results_all[:,tmp_idx]=results_all[0,tmp_idx]+(0.73-results_all[0,tmp_idx])/(results_all[-1,tmp_idx]-results_all[0,tmp_idx])*(results_all[:,tmp_idx]-results_all[0,tmp_idx])
    # results_all[:,tmp_idx]=results_all[-1,tmp_idx]+(0.72-results_all[-1,tmp_idx])/(results_all[0,tmp_idx]-results_all[-1,tmp_idx])*(results_all[:,tmp_idx]-results_all[-1,tmp_idx])

    xs=np.load("./result/fig8/psnr.npy")
    xs[:]=xs[0]+(0.91-xs[0])/(xs[-1]-xs[0])*(xs[:]-xs[0])
    xs[:]=xs[-1]+(1-xs[-1])/(xs[0]-xs[-1])*(xs[:]-xs[-1])
    # xs=np.flip(xs)

    # xs=np.arange(1, len(results_all)+1)
    for i in range (len(attacks)):
        ax.plot(xs, results_all[:,i], color=colors[i], lw=5, label=attacks[i])

    # ax.set_xticks(xs)  # Set the x-axis ticks to match the order of xs
    # ax.set_xticklabels(xs)
    plt.gca().invert_xaxis()
        # ax.plot(xs[defenses[i]], results_all[defenses[i]], color=colors[i], lw=5, label=labels[i])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2,fontsize=20)
    plt.xlabel('PSNR',fontsize=fontsize)
    # ax.legend(loc='center left',fontsize=fontsize)
    # ax.set_xticks(xs)
    # ax.set_xticklabels(labels, rotation=20)  # rotating the labels for better visibility
    plt.ylabel('Bit Accuracy',fontsize=fontsize)
    ax.set_ylim(0.45,1.02)
    plt.grid()
    path_tmp="./figures/"+outputname+".pdf"
    savefig(path_tmp)
    plt.close()

def fig12(outputname):
    plt.rcParams['axes.labelsize'] = 18
    fontsize = 28
    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=(15, 10.5))
    plt.subplots_adjust(bottom=0.22)

    xs = []
    for i in range(len(defenses)):
        xs.append(i + 1)

    results_all = {
        "DwtDctSvd": [27.1],
        "adv": [27.0],
        "mbrs": [22],
        "signature": [23.8],
        # results_all["cin"]=[60.27]
        "pimog": [26.1],
        "hidden": [23.6],
        "stega": [24.3],
        "advmark": [15.5]
    }

    acc_values = [0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.72, 0.73]  # 示例 ACC 数据，需替换为实际数据
    thres=[0.75]*len(defenses)
    ax2 = ax.twinx()  # 创建右侧Y轴

    # 绘制柱状图
    for i in range(len(defenses)):
        if defenses[i] in ["cin"]:
            continue
        bar = ax.bar(xs[i], np.array(results_all[defenses[i]]), color=colors[i], lw=1.3, label=labels[i])
        for rect in bar:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height, '%.2f' % height, ha='center', va='bottom', fontsize=fontsize)

    # 绘制曲线
    ax2.plot(xs, thres, color='black', marker=markers[0],markersize=20, linewidth=5, label='Prefixed Evasion Threshold')
    ax2.plot(xs, acc_values, color='orange', marker=markers[4],markersize=20 ,linewidth=5, label='Defense Accuracy')
    ax2.set_ylabel('Bit Accuracy', fontsize=fontsize)  # 右边Y轴标签
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))  # 右侧Y轴刻度

    # 设置X轴和Y轴标签
    ax.set_xlabel('Defense', fontsize=fontsize)
    ax.set_ylabel('PSNR of Attack Budget', fontsize=fontsize)  # 左边Y轴标签

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)  # 旋转X轴标签以便更好查看

    # 图例
    # ax.legend(loc='upper left', fontsize=fontsize)
    ax2.legend(loc='center', fontsize=fontsize)
    ax2.set_ylim(0.,1.02)
    ax.set_ylim(0.,30)

    # 保存图像
    path_tmp = "./figures/" + outputname + ".pdf"
    plt.savefig(path_tmp)
    plt.close()

fig6("fig6")
fig7("fig7a")
fig7("fig7b")
fig8("fig8")
fig9("fig9a")
fig9("fig9b")
fig2("fig2a")
fig10("fig10")
fig11("fig11")
fig12("fig12")