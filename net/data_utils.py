import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
BS=opt.bs
print(BS)
crop_size='all_img'
if opt.crop:
    crop_size=opt.crop_size


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

class DIV2kDataset_TRAIN(data.Dataset):
    def __init__(self,path='/home/zhilin007/VscodeProjects/dehaze/data',down='bicubic',flod=2,lr_size=48):
        super(DIV2kDataset_TRAIN,self).__init__()
        self.lr_size=lr_size
        self.flod=flod
        self.down=down
        self.train_imgs_dir=os.listdir(os.path.join(path,'DIV2K/DIV2K_train_HR'))
        self.hr_imgs=[os.path.join(path,'DIV2K/DIV2K_train_HR',img) for img in self.train_imgs_dir]
        self.lr_dir=os.path.join(path,f'DIV2K/DIV2K_train_LR_{down}/X{flod}/')
    def __getitem__(self, index):
        hr=Image.open(self.hr_imgs[index])
        # hr.show()
        img=self.train_imgs_dir[index]
        id=img.split('/')[-1].split('.')[0]
        lr_name=id+f'x{self.flod}'+'.png'
        # print(lr_name)
        lr=Image.open(self.lr_dir+lr_name)
        # lr.show()
        i,j,h,w=tfs.RandomCrop.get_params(lr,output_size=(self.lr_size,self.lr_size))
        # print(lr.size,i,j,h,w)
        hr=FF.crop(hr,i*self.flod,j*self.flod,h*self.flod,w*self.flod)
        lr=FF.crop(lr,i,j,h,w)
        
        # lr=hr.resize((48,48),Image.BICUBIC)
        # lr=Image.open(os.path.join(self.bicubic_dir,lr_name))
        # target.show()
        hr,lr=self.augData(hr.convert('RGB'),lr.convert('RGB'))
        return lr,hr
    def augData(self,data,target):
        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,1)
        rand_rot2=random.randint(0,1)
        rand_rot3=random.randint(0,1)
        data=tfs.RandomHorizontalFlip(rand_hor)(data)
        target=tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data=FF.rotate(data,90)
            target=FF.rotate(target,90)
        if rand_rot2:
            data=FF.rotate(data,180)
            target=FF.rotate(target,180)
        if rand_rot3:
            data=FF.rotate(data,270)
            target=FF.rotate(target,270)

        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.hr_imgs)

class Set5_Test(data.Dataset):
    def __init__(self,path='',down='bicubic',flod=2,lr_size=48):
        super(Set5_Test,self).__init__()
        self.lr_size=lr_size
        self.flod=flod
        self.down=down
        self.hr_dir=os.listdir(os.path.join(path,'Set5'))
        self.hr_imgs=[os.path.join(path,'Set5',img) for img in self.hr_dir]
    def __getitem__(self,index):
        hr=Image.open(self.hr_imgs[index])
        # print(hr.size)
        i,j,h,w=tfs.RandomCrop.get_params(hr,output_size=(self.lr_size*self.flod,self.lr_size*self.flod))
        hr=FF.crop(hr,i,j,h,w)
        # print(hr.size)
        lr=hr.resize((self.lr_size,self.lr_size),Image.BICUBIC)
        # print(lr.size)
        hr=tfs.ToTensor()(hr)
        lr=tfs.ToTensor()(lr)
        return lr,hr
    def __len__(self):
        return len(self.hr_imgs)


class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        # haze.show();print(haze.size)
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,20000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('_')[0]
        clear_name=id+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        # clear.show();print(clear.size)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            # print(i,j,h,w)
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        # print(haze.size())
        # print(clear.size())
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
           # rand_rot=random.randint(0,1)
           # rand_rot2=random.randint(0,1)
           # rand_rot3=random.randint(0,1)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
            
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)




import os
pwd=os.getcwd()
path='/home/zhilin007/VscodeProjects/dehaze/data'
path3='/home/nick/zl/dehaze/data'
path4='/home/peng/zl/dehaze/data'
path5='/home/User1/ninesun/dataset/'
if pwd.find('nick')!=-1 :
    path=path3
if pwd.find('peng')!=-1:
    path=path4
if pwd.find('ninesun')!=-1:
    path=path5


ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDEV0/ITS',train=True,size=crop_size),batch_size=BS,shuffle=True)
ITS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDEV0/SOTS/indoor',train=False,size='all img'),batch_size=1,shuffle=True)

OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDEV0/OTS',train=True,format='.jpg'),batch_size=BS,shuffle=True)
OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDEV0/SOTS/outdoor',train=False,size='all img',format='.png'),batch_size=1,shuffle=True)
#div_train_loader=DataLoader(dataset=DIV2kDataset_TRAIN(path) ,num_workers=4, batch_size=BS,shuffle=True)
#set5_test_loader=DataLoader(dataset=Set5_Test(path) ,num_workers=4, batch_size=1,shuffle=True)
if __name__ == "__main__":
    # loader = iter(DataLoader(dataset=DIV2kDataset_TRAIN() ,num_workers=1, batch_size=1,shuffle=True))
    # lr,hr=next(iter(OTS_test_loader_))
    # tensorShow([lr,hr],['',''])
    # iters=iter(ITS_test_loader)
    # print('loader length :',len(ITS_train_loader))
    # for i in range(600):
    #     data=next(iters)
    #     print('\r'+str(i),end='',flush=True)
    # tensorShow(tensors=(hr,lr),titles=["hr","lr"])
    # # print(lr.size())[8, 3, 96, 96]
    # im1=Image.open('/home/zhilin007/VscodeProjects/dehaze/net/111.png')
    # im2=Image.open('/home/zhilin007/VscodeProjects/dehaze/net/pred.png')
    # im1=tfs.ToTensor()(im1)
    # im2=tfs.ToTensor()(im2)
    # print(im2.size())
    # # loss=ssim(im1,im2)
    # print(im2)
    # loss=psnr(im1,im2) #float
    # print(loss)
    # print(loss.dtype)

# # lr to hr size
#     def lr_to_hr_size(lr,hr):
#         h=hr.size(3)
#         # print(lr.size())
#         lr=tfs.ToPILImage()(torch.squeeze(lr))
#         lr=lr.resize((h,h),Image.BICUBIC)
#         lr=tfs.ToTensor()(lr)[None,::]
#         # print(lr.size())
#         # print(hr.size())
#         return lr,hr
#     ssims=[]
#     psnrs=[]
#     i=0
#     while i < 30:
#         i+=1
#         lr,hr=lr_to_hr_size(*next(loader))
#         ssim1=ssim(hr,lr).item()
#         psnr1=psnr(hr,lr)
#         ssims.append(ssim1)
#         psnrs.append(psnr1)
    
#     print(np.mean(psnrs))
#     print(np.mean(ssims)) 
    pass
