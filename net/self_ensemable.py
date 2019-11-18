import torch 
import torchvision.transforms as tfs 
from torch import rot90,flip
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 
import numpy as np 
from metrics import *
from models import *
from data_utils import *
def tensorShow(tensors):
        fig=plt.figure()
        for tensor ,i in zip(tensors,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.axis('off')
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()
def tensor2ensemble(x):
	'''
	tensor:[1,3,h,w] -> ensemble :[8,3,h,w]
	'''
	_,c,w,h=x.size()
#flip :0 0   0   0   1  1  1  1
#rot  :0 90 180 270 0 90 180 270 
	x1=rot90(x,1,[-2,-1])
	x2=rot90(x,2,[-2,-1])
	x3=rot90(x,3,[-2,-1])
	x4=flip(x,[-1])
	x5=rot90(x4,1,[-2,-1])
	x6=rot90(x4,2,[-2,-1])
	x7=rot90(x4,3,[-2,-1])
	return [x,x1,x2,x3,x4,x5,x6,x7]
def ensemble2tensor(xs):
	x=xs[0]
	x1=rot90(xs[1],-1,[-2,-1])
	x2=rot90(xs[2],-2,[-2,-1])
	x3=rot90(xs[3],-3,[-2,-1])
	x4=flip(xs[4],[-1])
	x5=flip(rot90(xs[5],-1,[-2,-1]),[-1])
	x6=flip(rot90(xs[6],-2,[-2,-1]),[-1])
	x7=flip(rot90(xs[7],-3,[-2,-1]),[-1])
	x_out=(x+x1+x2+x3+x4+x5+x6+x7)/8.0
	return x_out


def ensemble_test(net):
	device='cuda' if torch.cuda.is_available() else 'cpu'
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	ssims2=[]
	i=0
	loader_test=ITS_test_loader_noSF #没有shuffle
	while i< 500 :
		print('\r'+str(i),end='',flush=True)
		i=i+1
		input,target=next(iter(loader_test))
		target=target.to(device)
		#ensemble_test
		inputs=tensor2ensemble(input)#element : 8 
		preds=[]
		for input in inputs:
			input=input.to(device)
			preds.append(net(input))
		pred=ensemble2tensor(preds)
		ssim1=ssim(pred,target).item()
		ssim2_=ssim2(pred,target).item()
		psnr1=psnr(pred,target)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		ssims2.append(ssim2_)
	print('for ensemble :')
	print(f'ssim:{np.mean(ssims):.6f}|ssim2:{np.mean(ssims2):.6f} | psnr:{np.mean(psnrs):.6f}')
	
def get_net(dir):
	ckp=torch.load(dir)
	net=RN829_1(rgb_mean=[0,0,0],gps=3,blocks=19)
	net=nn.DataParallel(net)
	net.to('cuda')
	net.load_state_dict(ckp['model'])
	net.eval()
	max_psnr=ckp['max_psnr']
	max_ssim=ckp['max_ssim']
	print(max_psnr,max_ssim,'at eval')
	return net 

if __name__ == "__main__":
	# im1=Image.open('net/test_imgs/1419_5.png')
	# im1=tfs.ToTensor()(im1)[None,::]
	# ims=tensor2ensemble(im1)
	# xs=ensemble2tensor(ims)
	# tensorShow([im1,xs])
	with torch.no_grad():
		dir='./trained_models/its_train_rn829_1_3_19.pk'
		net=get_net(dir)
		ensemble_test(net)

