import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()
        
        
class AFO_Dataset(data.Dataset):
    def __init__(self,path,train,size=240,format='.jpg'):
        super(AFO_Dataset,self).__init__()
        self.size=size
        self.train=train
        print('crop size',size)
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'Dark-images'))
        self.haze_imgs=[os.path.join(path,'Dark-images',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'GT-images')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
#         if isinstance(self.size,int):
#             while haze.size[0]<self.size or haze.size[1]<self.size :
#                 index=random.randint(0,20000)
#                 haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        clear_name=img.split('/')[-1]
        clear=Image.open(os.path.join(self.clear_dir,clear_name))

#         clear=tfs.CenterCrop(haze.size[::-1])(clear)
#         if not isinstance(self.size,str):
#         cropper=tfs.RandomCrop(size=(self.size,self.size))
#         haze=cropper(haze)
#         clear=cropper(clear)
#         if self.train:
        haze = haze.resize((456, 256))
        clear = clear.resize((456,256))
#             haze = tfs.CenterCrop(self.size)(haze)
#             clear = tfs.CenterCrop(self.size)(clear)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )

#         else:
#             haze = haze.resize((456, 256))
#             clear = clear.resize((456,256))
#             haze = tfs.CenterCrop(self.size)(haze)
#             clear = tfs.CenterCrop(self.size)(clear)
            
#             haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )

        return haze,clear, clear_name
    def augData(self,data,target):
#         if self.train:
#             rand_hor=random.randint(0,1)
#             rand_rot=random.randint(0,3)
#             data=tfs.RandomHorizontalFlip(rand_hor)(data)
#             target=tfs.RandomHorizontalFlip(rand_hor)(target)
#             if rand_rot:
#                 data=FF.rotate(data,90*rand_rot)
#                 target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
#         data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)