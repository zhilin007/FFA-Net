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
from self_ensemable import *
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
model_dir=abs+'trained_models/its_train_rn829_1_3_19.pk'
haze=abs+'test_imgs/1445_8.png'
# gt=abs+'test_imgs/1442.png'


forward_output=[]
forward_input=[]
def hook_func(module,input,output):
	# print(len(input))
	# print(input[0].size())
	forward_output.append(output.cpu().numpy())
	forward_input.append(input[0].cpu().numpy())

ckp=torch.load(model_dir,map_location='cpu')
net=RN829_1(rgb_mean=[0,0,0],gps=3,blocks=19)
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
device='cuda' if torch.cuda.is_available() else 'cpu'
net.module.palayer.register_forward_hook(hook_func)

def pred(net,haze1):
    with torch.no_grad():
        pred = net(haze1)
def ensemble_pred(net,haze1):
    with torch.no_grad():
        inputs=tensor2ensemble(haze1)#element : 8 
        preds=[]
        for input in inputs:
            input=input.to(device)
            preds.append(net(input))
        pred=ensemble2tensor(preds)

# res = -1*(pred-haze1)

# print(len(forward_input))
# forward_input=forward_input[0]
# forward_output=forward_output[0]
# np.save('pa_input_1419.npy',forward_input)
# np.save('pa_output_1419.npy',forward_output)




# res=torch.squeeze(res)
# res=res-torch.min(res)
# res=res/torch.max(res)

# p1=tfs.ToPILImage()(res).convert('RGB')
# p1.show()
# p2=p1.convert('L')
# p2.show()
ts=torch.squeeze(pred.clamp(0,1).cpu())
# tensorShow([haze_no,gt,pred.clamp(0,1),res],['haze','gt','pred','res'])
# ts=vutils.make_grid([torch.squeeze(haze_no.cpu()),torch.squeeze(gt.cpu()),torch.squeeze(pred.clamp(0,1).cpu()),torch.squeeze(res.cpu())])
# vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
vutils.save_image(ts,'1445_ffa.png')
