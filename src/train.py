import os,sys
import time,math
import random
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch,warnings
from torch import nn
import torchvision
import torchvision.utils as vutils
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
warnings.filterwarnings('ignore')
from torchvision.models import vgg16
from model import *
from losses import *
from PerceptualLoss import PerLoss
from metrics import psnr,ssim
from torch.backends import cudnn
from torch import optim
from dataset import AFO_Dataset


# define function for argparser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model_wts', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--log_dir', type=str, default='../logs')

    parser.add_argument('--perloss', type=bool, default=True)
    parser.add_argument('--edgeloss', type=bool, default=True)
    parser.add_argument('--fftloss', type=bool, default=True)

    parser.add_argument('--groups', type=int, default=3)
    parser.add_argument('--blocks', type=int, default=16)

    return parser.parse_args()


def lr_schedule_cosdecay(t,T,init_lr=1e-4):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def train(net,train_loader,test_loader,optim,criterion,args):
    losses=[]
    start_step=0
    T=args.epochs
    max_ssim=0
    max_psnr=0
    ssims=[]
    psnrs=[]
    if args.resume:
        print(f'resume from {args.model_wts}')
        ckp=torch.load(args.model_wts)
        losses=ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step=ckp['step']
        max_ssim=ckp['max_ssim']
        max_psnr=ckp['max_psnr']
        psnrs=ckp['psnrs']
        ssims=ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else :
        print('train from scratch *** ')
    start_time=time.time()    
    for step in range(start_step+1,T+1):
        for x, y, _ in tqdm(train_loader):
            net.train()
            lr=1e-4
            lr=lr_schedule_cosdecay(step,T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr  
                
            x=x.to(device);y=y.to(device)
            out=net(x)
            loss=criterion[0](out,y)
            if perloss:
                loss2=criterion[1](out,y)
                loss=loss+0.04*loss2
            if edgeloss:
                loss3 = criterion[2](out, y)
                loss = loss+loss3
            if fftloss:
                loss4 = criterion[3](out, y)
                loss = loss + 0.01*loss4

    
            loss.backward()

            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
        print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{T}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

#         with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
#         	writer.add_scalar('data/loss',loss,step)

        if step % 1 ==0 :
            with torch.no_grad():
                ssim_eval,psnr_eval=test(net,test_loader)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
            # 	writer.add_scalar('data/ssim',ssim_eval,step)
            # 	writer.add_scalar('data/psnr',psnr_eval,step)
            # 	writer.add_scalars('group',{
            # 		'ssim':ssim_eval,
            # 		'psnr':psnr_eval,
            # 		'loss':loss
            # 	},step)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if psnr_eval > max_psnr :
                max_ssim=max(max_ssim,ssim_eval)
                max_psnr=max(max_psnr,psnr_eval)
                torch.save({
                            'step':step,
                            'max_psnr':max_psnr,
                            'max_ssim':max_ssim,
                            'ssims':ssims,
                            'psnrs':psnrs,
                            'losses':losses,
                            'model':net.state_dict()
                },f'{args.log_dir}/LPEF_Epoch{step}.pth')   #LPEF signifies model trained with L1+perceptual loss+edge loss+fft loss
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

def test(net,test_loader):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	#s=True
	for i ,(inputs,targets, _) in enumerate(test_loader):
		inputs=inputs.to(device);targets=targets.to(device)
		pred=net(inputs)
		# tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
		# vutils.save_image(targets.cpu(),'target.png')
		# vutils.save_image(pred.cpu(),'pred.png')
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		#if (psnr1>max_psnr or ssim1 > max_ssim) and s :
		#		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
		#		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
		#		s=False
	return np.mean(ssims) ,np.mean(psnrs)



if __name__=="__main__":

    args = parse_args()
    
    path = f'{args.data_dir}/train'#path to your 'data' folder
    train_data = AFO_Dataset(path,train=True, size=200) # size here refers to the crop_size
    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True)

    path = f'{args.data_dir}/test'#path to your 'data' folder
    test_data = AFO_Dataset(path,train=False, size=200)
    test_loader=DataLoader(test_data,batch_size=args.batch_size,shuffle=False)

    # fig = plt.figure(figsize = (20,20))
    # for i in range(4):
    #     plt.subplot(4, 4, 2*i + 1)
    #     n = np.random.randint(len(train_data))
    #     plt.imshow(train_data[n][0].permute(1,2,0).numpy())
    #     plt.subplot(4, 4, 2*i + 2)
    #     plt.imshow(train_data[n][1].permute(1,2,0).numpy())

    # plt.show()

    #################
    ## Train Model###
    #################
    models_={
        'lvrnet':LVRNet(gps=args.groups,blocks=args.blocks),
    }
    loaders_={
        'train':train_loader,
        'test':test_loader
    }

    net=models_['lvrnet']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net=net.to(device)
    if device=='cuda':
        net=torch.nn.DataParallel(net)
        cudnn.benchmark=True

    perloss = args.perloss
    edgeloss = args.edgeloss
    fftloss = args.fftloss
    criterion = []
    criterion.append(nn.L1Loss().to(device))
    if perloss:
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        criterion.append(PerLoss(vgg_model).to(device))
    if edgeloss:
        criterion.append(EdgeLoss())
    if fftloss:
        criterion.append(fftLoss())

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=1e-4, betas = (0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()

    train(net,train_loader,test_loader,optimizer,criterion,args)
