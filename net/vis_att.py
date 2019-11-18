import torch
from torchvision import transforms
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt  
from torchvision.utils import make_grid

def np_plot(x,labels=['channel','group']):
	# print(x.shape[0])
	n=x.shape[0]
	for i in range(n):
		ax=plt.subplot(1,1,1+i)
		ax.axis('off')
		ax.imshow(x[i])
	
	plt.show() 
def CA():
	ca=np.load('channelAtten_1419.npy')
	print(ca.shape)
	print(f'M1:M2:M3={np.sum(ca,axis=-1)}')#M1:M2:M3=[29.378365 30.288967 30.96829 ]
	print(ca)
	# np_plot(ca)
def PA():
	pa_input=np.squeeze(np.load('pa_input_1419.npy'))
	pa_output=np.squeeze(np.load('pa_output_1419.npy'))
	print(pa_input.shape)
	print(pa_output.shape)
	pa=pa_output/pa_input
	print(pa.shape)
	# print(pa[0]==pa[1])
	np_plot(pa[[0]])
CA()
