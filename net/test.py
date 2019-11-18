import os
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
abs=os.getcwd()+'/net/'
def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

model_dir=abs+'trained_models/its_train_ffa_3_19.pk'
haze=abs+'test_imgs/1445_8.png'

# gt=abs+'test_imgs/1442.png'

device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=FFA(gps=3,blocks=19)
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
haze = Image.open(haze)
# gt=Image.open(gt)
haze1= tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
])(haze)[None,::]
haze_no=tfs.ToTensor()(haze)[None,::]
# gt=tfs.ToTensor()(gt)[None,::]

def pred(net,haze1):
    with torch.no_grad():
        pred = net(haze1)

ts=torch.squeeze(pred.clamp(0,1).cpu())
tensorShow([haze_no,pred.clamp(0,1)],['haze','pred'])
# ts=vutils.make_grid([torch.squeeze(haze_no.cpu()),torch.squeeze(gt.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
# vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
vutils.save_image(ts,'1445_ffa.png')
