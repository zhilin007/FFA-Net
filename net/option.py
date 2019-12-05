import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=100000)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=bool,default=True)
parser.add_argument('--eval_step',type=int,default=5000)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='./trained_models/')
parser.add_argument('--trainset',type=str,default='its_train')
parser.add_argument('--testset',type=str,default='its_test')
parser.add_argument('--net',type=str,default='ffa')
parser.add_argument('--gps',type=int,default=3,help='residual_groups')
parser.add_argument('--blocks',type=int,default=20,help='residual_blocks')
parser.add_argument('--bs',type=int,default=16,help='batch size')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=240,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss')

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+str(opt.gps)+'_'+str(opt.blocks)
opt.model_dir=opt.model_dir+model_name+'.pk'
log_dir='logs/'+model_name

print(opt)
print('model_dir:',opt.model_dir)


if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
