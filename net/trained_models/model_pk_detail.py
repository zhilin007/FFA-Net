import torch
import os

def print_ckp(dir):
	ck=torch.load(dir,map_location='cpu')
	for key,value in ck.items():
		if key=='model':
			print(f'model : {dir}')
		elif key == 'ssims' or key =='psnrs' or key =='losses':
			print(f'{key}[-1] : {value[-1]}')
		else :
			print(f'{key} : {value}')
			

if __name__ == "__main__":
	dir='ots_train_rn829_1_3_19.pk'
	print_ckp(dir)
'''
before
step : 292750
max_psnr : 35.76263943009184
max_ssim : 0.9846133173236091
ssims[-1] : 0.9837651610231957
psnrs[-1] : 35.76263943009184
losses[-1] : 0.011336984112858772
model : its_train_rn829_1_3_19.pk

step : 235750
max_psnr : 33.37671316946244
max_ssim : 0.9803887207391283
ssims[-1] : 0.9791293846139782
psnrs[-1] : 33.37671316946244
losses[-1] : 0.018336495384573936
model : ots_train_rn829_1_3_19.pk


'''
