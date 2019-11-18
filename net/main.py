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
print('log_dir :',log_dir)

print('model_name:',model_name)


mean=[0.45627367, 0.38360259, 0.36426624]#clear
models_={
	'net':Net(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	'net_fa':Net_FA(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),

	# 'rcan':RCAN(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rcan821': RCAN_modify(),
	# 'rcan823_1':RCAN_DiConv(mean,),
	# 'rcan822_2':RPAN(mean),
	# 'rcan824_1':RCAN824_1(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rpan824_2':RPAN824_2(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rpcan824_3':RPCAN824_3(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rcan824_5':RCAN824_5(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rcan824_7':RCAN824_7(gps=opt.gps,blocks=opt.blocks),
	# 'rcan824_6':RCAN(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rcan824_fusion1_g':RCAN824_fusion1_G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rcan824_fusion2_g':RCAN824_fusion2_G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),

	# 'rcan825_d1':RCAN825_D1(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rcan825_d2':RCAN825_D2(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rpan825_2':RPAN825_2(gps=opt.gps,blocks=opt.blocks),
	# 'rpan825_f2g':RPAN825_F2G(gps=opt.gps,blocks=opt.blocks),
	# 'rpan825_d1':RPAN825_D1(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rpan825_d2':RPAN825_D2(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rpan825_d2_f2g':RPAN825_D2_F2G(gps=opt.gps,blocks=opt.blocks),
	# 'rcan_perloss':RCAN(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),

	# 'rcan826_d3':RCAN826_D3(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),

	# 'rn826_d1_ca_pa':RN826_D1_CA_PA(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rn826_d1_ca_pa_f2g':RN826_D1_CA_PA_F2G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rn826_d1_ca_f2g':RN826_D1_CA_F2G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rn826_d1_ca_pa_f3g':RN826_D1_CA_PA_F3G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rn826_d1_ca_pa_f4g':RN826_D1_CA_PA_F4G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),

	# 'rn827_d3_f2g':RN827_D3_F2G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rcan827_1':RCAN827_1(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rn827_d1_ca_f2g_2':RN827_D1_CA_F2G_2(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rcan827_diconv':RCAN827_DiConv(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rn827_d1_ca_f2g':RN827_D1_CA_F2G(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rn829_1':RN829_1(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	# 'rn829_2':RN829_2(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks),
	'ffa':FFA(gps=opt.gps,blocks=opt.blocks),
	# #'rn_final':RN_Final(rgb_mean=mean,gps=opt.gps,blocks=opt.blocks)
}
loaders_={
	#'DIV':div_train_loader,
	#'set5':set5_test_loader,
	'its_train':ITS_train_loader,
	'its_test':ITS_test_loader,
	#'ots_train':OTS_train_loader,
	#'ots_test':OTS_test_loader
}
start_time=time.time()
T=opt.steps	
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,loader_test,optim,criterion):
	losses=[]
	start_step=0
	max_ssim=0
	max_psnr=0
	ssims=[]
	psnrs=[]
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

		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')
	# print(net)
	# print(start_step,opt.steps)
	for step in range(start_step+1,opt.steps+1):
		#print(step)
		net.train()
		lr=opt.lr
		if not opt.no_lr_sche:
			lr=lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr  
		x,y=next(iter(loader_train))
		# print(x.shape,y)

		x=x.to(opt.device);y=y.to(opt.device)
		out=net(x)
	
		loss=criterion[0](out,y)
		if opt.perloss:
			loss2=criterion[1](out,y)
			loss=loss+0.04*loss2
		loss=loss/opt.m*1.0
		loss.backward()
		if step%opt.m == 0 :
			optim.step()
			optim.zero_grad()
			losses.append(loss.item())
			print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)
		
		with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
			writer.add_scalar('data/loss',loss,step)

		if step % opt.eval_step ==0 :
			with torch.no_grad():
				ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,step)
				
			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
				writer.add_scalar('data/ssim',ssim_eval,step)
				writer.add_scalar('data/psnr',psnr_eval,step)
				writer.add_scalars('group',{
					'ssim':ssim_eval,
					'psnr':psnr_eval,
					'loss':loss
				},step)

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			if ssim_eval > max_ssim or psnr_eval > max_psnr :
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
				},opt.model_dir)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

	np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy',losses)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy',ssims)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)


def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	
	
	s=True
	
	for i ,(inputs,targets) in enumerate(loader_test):
		
		
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		pred=net(inputs)
		# # print(pred)
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


if __name__ == "__main__":

	loader_train=loaders_[opt.trainset]
	loader_test=loaders_[opt.testset]
	net=models_[opt.net]
	if opt.MSRAinit:
		print('-----MSRAinit-----')
		net.apply(weights_init) #!!!convinit * 0.1
	net=net.to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.perloss:
			vgg_model = vgg16(pretrained=True).features[:16]
			vgg_model = vgg_model.to(opt.device)
			for param in vgg_model.parameters():
				param.requires_grad = False
			criterion.append(PerLoss(vgg_model).to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	# optimizer = optim.SGD(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
	train(net,loader_train,loader_test,optimizer,criterion)
	

