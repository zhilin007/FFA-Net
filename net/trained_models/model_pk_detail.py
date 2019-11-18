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
	dir='***.pk'
	print_ckp(dir)

