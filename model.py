import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import cv2
from PIL import Image
import random
import torch.nn.functional as F
import os
from tqdm import tqdm
import gc
from videoloader import videoloader 
from convolutional_lstm import CLSTM
try:
	os.mkdir('model')
except:
	pass
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class model(nn.Module):
	def __init__(self,num_features,filter_size,height,width,channels,num_layers,batch_size):
		super(model, self).__init__()
		self.clstm1=CLSTM((height,width),channels,filter_size,num_features,num_layers)
		
		self.clstm1=self.clstm1.cuda()

		self.clstm1.apply(weights_init)
		self.batch_size=batch_size
		
		
		self.clstm2=CLSTM((height,width),2*channels+2*num_features,filter_size,num_features,num_layers)
		
		self.clstm2=self.clstm2.cuda()
		
		self.clstm2.apply(weights_init)

	def stackeven(self,ip):
		stacked=torch.zeros(ip.size()[0]//2,ip.size()[1],2*ip.size()[2],ip.size()[3],ip.size()[4]).cuda()
		buff=[]
		stackedindex=0
		for i in range(ip[1].size()[0]):
			buff.append(ip[i,:,:,:,:])
			if len(buff)==3:
				stacked[stackedindex,:,:ip.size()[2],:,:]=buff[0]
				stacked[stackedindex,:,:ip.size()[2],:,:]=buff[2]
				temp=buff[2].copy()
				buff=[]
				buf.append(temp)
		return stacked
	def init_hidden1(self,known,states=None):
		if known==1:
			self.clstm1.init_hidden_known(states)
		else:
			self.clstm1.init_hidden()
	def init_hidden2(self,known,states=None):
		if known==1:
			self.clstm2.init_hidden_known(states)
		else:
			self.clstm2.init_hidden()

	def encoder(self,x):
		b=torch.split(x,x.size()//50,dim=0)
		b=list(b)
		self.init_hidden1(0)
		h=[]
		for k in b:
			temp=self.clstm1(k)
			h.append(temp)
			self.init_hidden1(temp)
		out=torch.stack(h,dim=0)
		return out



	def forward(self,x,y):

		#print(x.size())
		self.clstm1.init_hidden(self.batch_size)
		a=self.clstm1(x)
		self.clstm2.init_hidden_known(a[0])
		
		stacked=self.stackeven(a[1])

		y[:,:,x.size()[2]:x.size()[2]+2*a[1].size(2),:,:]=stacked
		
		b=self.clstm2(y)
		

		return a,b


#b[0]:states
class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.conv1=nn.Conv2d(6,6,3,stride=1,padding=1)
		self.conv2=nn.Conv2d(6,3,3,stride=1,padding=1)
		self.conv3=nn.Conv2d(3,1,3,stride=1,padding=1)
		self.fc1=nn.Linear(320*240,1)
	def forward(self,x):
		x=F.elu(self.conv1(x))
		x=F.elu(self.conv2(x))
		x=F.elu(self.conv3(x))
		x=x.view(1,320*240)
		x=F.sigmoid(self.fc1(x))
		return x

# 0 1 2 3 4 5 6 7 8 9 10 11
from tqdm import tqdm
numepochs=1000
mod=model(6,3,320,240,6,5,1)
d=discriminator()
d=d.cuda()
mod = mod.cuda()
mod.load_state_dict(torch.load('model/model.hd5'))
d.load_state_dict(torch.load('model/dis'))
criterion = nn.L1Loss().cuda()
criterion1=nn.MSELoss().cuda()
criterion2=nn.BCELoss().cuda()
optimizer=optim.Adam(mod.parameters(),lr=1e-2)
doptimizer=optim.Adam(d.parameters(),lr=1e-5)
data=videoloader('processed_data',3)
for x1 in range(numepochs):
	l=0
	for j in tqdm(range(len(data))):
		

		out=data[j]
		if out==None:
			continue
		(x,y,x_tag,y_tag)=out
		
		predicted_x,predicted_y=mod(x,y)
		#predicted_y[1]=predicted_y[1]*trainableconstant
		d.zero_grad()

		###print(y_tag.size())
		temp1=y_tag.detach()
		temp2=predicted_y[1].detach()
		for i in range(temp1.size()[0]):
			#temp1=y_tag[i,:,:,:,:].clone()
			disoutreal=d(temp1[i,:,:,:,:])
			realloss=criterion2(disoutreal,Variable(torch.ones(1)).cuda())
			realloss.backward()
			#temp2=predicted_y[1][i,:,:,:,:].clone()
			disoutfake=d(temp2[i,:,:,:,:])
			fakeloss=criterion2(disoutfake,Variable(torch.zeros(1)).cuda())
			fakeloss.backward()
		doptimizer.step()
		mod.zero_grad()
		lossgan=0
		for i in range(y_tag.size()[0]):
			gendisout=d(predicted_y[1][i,:,:,:])
			gendisloss=criterion2(gendisout,Variable(torch.ones(1)).cuda())
			lossgan=lossgan+gendisloss


		#print(predicted_y[2].size())
		#print(predicted_x[1].size())
		#print(x_tag.size())
		#print(y_tag.size())
		
		loss=torch.add(0.6*torch.add(criterion(predicted_y[1],y_tag),criterion(predicted_x[1],x_tag)),0.001*torch.add(criterion1(predicted_y[1],y_tag),criterion1(predicted_x[1],x_tag)))+0.0*lossgan
		
		odd=0
		ycnt=0
		if x1%5==0:
			for f in range(x.data.size()[0]):
				if odd%2==0:
					curimage=data.convert2dtocomplex(x[f,:,:,:,:])
					cv2.imwrite('image'+str(f)+'.jpg',curimage)
					#cv2.waitKey(500)
				else:
					curimage=data.convert2dtocomplex(predicted_y[1].data[ycnt,:,:,:,:])
					#print(curimage)
					cv2.imwrite('image'+str(f)+'.jpg',curimage)
					#cv2.waitKey(100)
					ycnt+=1
				odd+=1
			#cv2.destroyAllWindows()
		


		
		loss.backward()
		l+=loss.data[0]/len(data)
		
		optimizer.step()
	torch.save(d.state_dict(),'model/dis')
	torch.save(mod.state_dict(),'model/model.hd5')
	print(str(x1)+':'+str(l))


