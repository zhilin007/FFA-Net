import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16

'''
python test_dataset.py --net='rn829_1' --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0002 --trainset='ots_train' --testset='ots_test'
python test_dataset.py --net='rn829_1' --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0002 --m=2
'''

mean=[0.45627367, 0.38360259, 0.36426624]#clear
models_={
	'rn827_d1_ca_f2g':RN827_D1_CA_F2G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	'rn829_1':RN829_1(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),


}
loaders_={
	'its_train':ITS_train_loader,
	'its_test':ITS_test_loader,
	'ots_train':OTS_train_loader,
	'ots_test':OTS_test_loader
}
def reeval(net,loader_train,loader_test):
	print(f'\n --------model : :{opt.model_dir} -----------')
	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp=torch.load(opt.model_dir)
		losses=ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step=ckp['step']
		max_ssim=ckp['max_ssim']
		max_psnr=ckp['max_psnr']
		psnrs=ckp['psnrs']
		ssims=ckp['ssims']
		print('之前：')
		print('max_ssim:',max_ssim,'max_psnr:',max_psnr)
		print('ssim[-1]:',ssims[-1],'psnrs[-1]:',psnrs[-1])
		print(f'start_step:{start_step} start evaling ---')
	else :
		raise RuntimeError('no trained model')

	with torch.no_grad():
		ssim_eval,psnr_eval=test(net,loader_test)
		print(f'ssim:{ssim_eval:.4f} | psnr:{psnr_eval:.4f}')
		max_ssim=max(max_ssim,ssim_eval)
		max_psnr=psnr_eval #这里需要修改
		torch.save({
					'step':start_step,
					'max_psnr':max_psnr,
					'max_ssim':max_ssim,
					'ssims':ssims,
					'psnrs':psnrs,
					'losses':losses,
					'model':net.state_dict()
		},opt.model_dir)
		print(f'\n ------ model saved at step :{start_step}| reTest_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}  -------')

def test(net,loader_test):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	l=len(loader_test)
	for i ,(inputs,targets) in enumerate(loader_test):
		print(f'\r eval:{i}/{l}',end='',flush=True)
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		pred=net(inputs)
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	return np.mean(ssims) ,np.mean(psnrs)



if __name__ == "__main__":
	loader_train=loaders_[opt.trainset]
	loader_test=loaders_[opt.testset]
	net=models_[opt.net]
	
	net=net.to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	reeval(net,loader_train,loader_test)
	

