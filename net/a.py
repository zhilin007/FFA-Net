import os
import numpy as np
from PIL import Image
from models import *
import torch
import torchvision.transforms as tfs 
import os
import torchvision.utils as vutils
abs=os.getcwd()+'/net/'
dir=abs+'its_train_rn827_d1_ca_f2g_3_19.pk'

ckp=torch.load(dir,map_location='cpu')
net=RN827_D1_CA_F2G(rgb_mean=[0,0,0],gps=3,blocks=19)
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])

torch.save({
							'step':0,
							'max_psnr':0,
							'max_ssim':0,
							'ssims':[],
							'psnrs':[],
							'losses':[],
							'model':net.state_dict()
				},'model.pk')