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
try:
	os.mkdir('model')
except:
	pass
class model(nn.Module):
	def __init__(self):
		super(model, self).__init__()
		self.model_ft = models.resnet18(pretrained=True)
		self.res50_conv = nn.Sequential(*list(self.model_ft.children())[:-2])
		for child in self.model_ft.children():
			for param in child.parameters():
				param.requires_grad=False
		#print(self.model_ft)
		self.conv11=nn.Conv2d(512,1,kernel_size=3,stride=1)
		self.batch_norm_conv11=nn.InstanceNorm2d(1)
		
		self.conv12=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
		self.batch_norm_conv12=nn.InstanceNorm2d(1)
		
		self.conv21=nn.Conv2d(512,1,kernel_size=3,stride=1)
		self.batch_norm_conv21=nn.InstanceNorm2d(1)

		self.conv22=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
		self.batch_norm_conv22=nn.InstanceNorm2d(1)

		self.conv31=nn.Conv2d(1,16,kernel_size=9,stride=1)
		self.batch_norm_conv31=nn.InstanceNorm2d(16)
		
		self.conv41=nn.Conv2d(16,1,kernel_size=3,stride=1)
		self.batch_norm_conv41=nn.InstanceNorm2d(1)

		self.conv32=nn.Conv2d(1,16,kernel_size=9,stride=1)
		self.batch_norm_conv32=nn.InstanceNorm2d(16)
		self.conv42=nn.Conv2d(16,1,kernel_size=3,stride=1)
		self.batch_norm_conv42=nn.InstanceNorm2d(1)

		self.conv33=nn.Conv2d(1,16,kernel_size=9,stride=1)
		self.batch_norm_conv33=nn.InstanceNorm2d(16)
		self.conv43=nn.Conv2d(16,1,kernel_size=3,stride=1)
		self.batch_norm_conv43=nn.InstanceNorm2d(1)

		self.conv34=nn.Conv2d(1,16,kernel_size=9,stride=1)
		self.batch_norm_conv34=nn.InstanceNorm2d(16)
		self.conv44=nn.Conv2d(16,1,kernel_size=3,stride=1)
		self.batch_norm_conv44=nn.InstanceNorm2d(1)

		self.deconv1=nn.ConvTranspose2d(1,3,10,stride=4)
		self.batch_norm_deconv1=nn.InstanceNorm2d(3)
		self.deconv2=nn.ConvTranspose2d(3,3,11,stride=4)
		self.batch_norm_deconv2=nn.InstanceNorm2d(3)
		self.deconv3= nn.ConvTranspose2d(3,3,4,stride=2)

		#self.deconv4=nn.ConvTranspose2d(3,3,11,stride=4)
		#self.deconv5=nn.ConvTranspose2d(3,3,2,stride=2)
		self.scaler = transforms.Scale((224, 224))
		self.to_tensor = transforms.ToTensor()
	def forward(self,x,y,c_state,h_state):
		#print("hheloo")
		one=Variable(torch.zeros(1,3,224,224),requires_grad=False).cuda()
		two=Variable(torch.zeros(1,3,224,224),requires_grad=False).cuda()
		one[0,:,:,:]=self.to_tensor(self.scaler(x)).cuda()
		two[0,:,:,:]=self.to_tensor(self.scaler(y)).cuda()
		one=one/255.0
		two=two/250.0
		one1=one.view(1,224,224,3)
		two1=two.view(1,224,224,3)
		one=F.relu(self.res50_conv(one))
		two=F.relu(self.res50_conv(two))
		one=F.relu(self.conv12(F.relu(self.conv11(one))))
		two=F.relu(self.conv22(F.relu(self.conv21(two))))
		one=one.view(5,5)
		two=two.view(5,5)
		h_state0=torch.cat((one,h_state,two),1)
		temp_h_state=h_state.clone()
		temp_c_state=c_state.clone()
		
		
		
		zsf=torch.zeros(10,15).cuda()
		h_state11=torch.cat((h_state0,zsf),0)
		h_state12=h_state11.view(1,1,15,15)
		h_state13=F.relu(self.conv31(h_state12))
		f=F.sigmoid(self.conv41(h_state13))
		
		zsi=torch.zeros(10,15).cuda()
		h_state21=torch.cat((h_state0,zsi),0)
		h_state22=h_state21.view(1,1,15,15)
		h_state23=F.relu(self.conv32(h_state22))
		i=F.tanh(self.conv42(h_state23))

		zsc=torch.zeros(10,15).cuda()
		h_state31=torch.cat((h_state0,zsc),0)
		h_state32=h_state31.view(1,1,15,15)
		h_state33=F.relu(self.conv33(h_state32))
		c=F.sigmoid(self.conv43(h_state33))

		zso=torch.zeros(10,15).cuda()
		h_state41=torch.cat((h_state0,zso),0)
		h_state42=h_state41.view(1,1,15,15)
		h_state43=F.relu(self.conv34(h_state42))
		o=F.sigmoid(self.conv44(h_state43))

		c_state=torch.mul(f,temp_h_state)+torch.mul(i,c)

		h_state=torch.mul(o,F.tanh(c_state))

		out1=F.relu(self.deconv1(h_state))
		out2=F.relu(self.deconv2(out1))
		out5=self.deconv3(out2)
		#out4=F.relu(self.deconv4(out3))
		#out5=self.deconv5(out4)

		
		
		return h_state,c_state,one1[0,:,:,:],out5[0,:,:,:].view(224,224,3),two1[0,:,:,:],out5[0,:,:,:]

import os
torch.cuda.device(1)
network=model()
network.cuda()
network.load_state_dict(torch.load('model/model'))
#torch.backends.cudnn.enabled=False"""
batch_size=2
optimizer=optim.Adam(network.parameters(),lr=1e-4)
dirs=os.listdir('processed_data')
dirs=dirs[:20]
train_data=[]

for w in range(10000):
	total_loss=0
	show_vid=None
	for k in tqdm(range(int(len(dirs)/batch_size))):
		#optimizer.zero_grad()
		batch_dirs=[]
		batch_dirs=dirs[k*batch_size:k*batch_size+batch_size]
		first_frames=[]
		second_frames=[]
		target_frames=[]
		for k1 in batch_dirs:
			files=os.listdir('processed_data/'+k1)
			files.sort()
			i=0
			cur_frame1=[]
			cur_frame2=[]
			cur_target_frames=[]
			while i<len(files):
				try:
					fr1=Image.open('processed_data/'+k1+'/'+files[i])

				except:
					break
				if i+1>=len(files):
					break
				try:
					fr2=Image.open('processed_data/'+k1+'/'+files[i+1])
				except:
					break
				fr3=None
				if i+2>=len(files) or i+2==len(files)-1:
					fr3=Image.new('RGB',(320,240))
				else:
					try:
						fr3=Image.open('processed_data/'+k1+'/'+files[i+2])
					except:
						fr3=Image.new('RGB',(320,240))

				cur_frame1.append(fr1)
				cur_frame2.append(fr3)
				fr2=network.to_tensor(network.scaler(fr2)).cuda()
				cur_target_frames.append(fr2)
				i+=2
			first_frames.append(cur_frame1)
			second_frames.append(cur_frame2)
			target_frames.append(cur_target_frames)
		train_data.append([first_frames,second_frames,target_frames])
		#cur_state=Variable(torch.randn(len(first_frames),5,5),requires_grad=False).cuda()
		cnt=0
		train_frames=[]
		all_videos=[]
		for i,j,l in zip(first_frames,second_frames,target_frames):
			
			numlists1=[]
			numlists2=[]
			num_target_frames=[]
			for z in range(int(len(i)/100)):
				numlists1.append([])
				numlists2.append([])
				num_target_frames.append([])
			if len(i)%100!=0:
				numlists1.append([])
				numlists2.append([])
				num_target_frames.append([])
			g1=0
			g2=0
			for z in i:
				g2+=1
				numlists1[g1].append(z)
				if g2%100==0:
					g1+=1 
			g1=0
			g2=0
			for z in j:
				g2+=1
				numlists2[g1].append(z)
				if g2%100==0:
					g1+=1 
			g1=0
			g2=0
			for z in l:
				g2+=1
				num_target_frames[g1].append(z)
				if g2%100==0:
					g1+=1 


			c_state=torch.zeros(1,5,5).cuda()
			h_state=torch.zeros(1,5,5).cuda()
			hidden=None
			c_hidden=None
			for i1,j1,l1 in zip(numlists1,numlists2,num_target_frames):
				entire_video=[]
				video_for_train=[]
				batch_loss=torch.zeros(1).cuda()
				if hidden is not None:
					h_state=hidden
				if c_hidden is not None:
					c_state=c_hidden
				optimizer.zero_grad()
				for q in range(len(i1)):
					h_state[0,:,:],c_state[0,:,:],a,b,c,d=network(i1[q],j1[q],c_state[0,:,:],h_state[0,:,:])
					entire_video.append(a)
					entire_video.append(b*255)
					entire_video.append(c)
					l1[q]=l1[q]/255.0
					mseloss=torch.mean(torch.abs(d-l1[q]))
					print(mseloss)
					#print(torch.mean(d))
					#print(torch.mean(l1[q]))
					batch_loss=torch.add(batch_loss,torch.mean(mseloss))
				all_videos.append(entire_video)
				hidden=h_state.detach()
				c_hidden=c_state.detach()
				batch_loss.backward()	
				optimizer.step()

				total_loss+=batch_loss.data


				
			cnt+=1
		total_loss/=int(len(dirs)/batch_size)
		u=random.randint(0,len(all_videos)-1)
		show_vid=all_videos[u]
		torch.cuda.empty_cache()
	for s in show_vid:
		s=s.data.cpu().numpy()
		#print(s.shape)
		cv2.imshow('image',s)
		cv2.waitKey(1)
	cv2.destroyAllWindows()
	torch.save(network.state_dict(),'model/model')
	gc.collect()
	
	print(str(w)+' '+str(total_loss))
import pickle
print(len(train_data))
with open('data.pickle','wb') as f:
    pickle.dump(train_data,f)

