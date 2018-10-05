import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
class videoloader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir,filter_size, transform=None):
        videos=os.listdir(root_dir)
        self.root_dir=root_dir
        self.data=videos



  

    def convert2dtocomplex(self,x):
        #print(x.size())
        outimag=torch.zeros(3,x.size()[2],x.size()[3])
        outreal=torch.zeros(3,x.size()[2],x.size()[3])
        outimag=x[0,3:,:,:]
        #print(outimag.size())

        outreal=x[0,:3,:,:]
        outimag=outimag.cpu().numpy()
        outreal=outreal.cpu().numpy()
        outimag=outimag.astype('complex')
        iota=np.array([1j])
        outimag=outimag*iota

        #print(outreal.size())
        out=outreal+outimag
        out=np.reshape(out,(out.shape[2],out.shape[1],3))
        #out=np.transpose(out,(1,2,0))
        #print(out.shape)
        outifft=np.fft.ifft(out)*800
        #print(outifft)
        outifft=outifft.astype('float64')
        #finalout=np.reshape(finalout,(x.size()[2],x.size()[3],3))
        return outifft
    def __len__(self):
        return len(self.data)
    def convertcomplexto2d(self,data):
        out=np.zeros((2*data.shape[0],data.shape[1],data.shape[2]),dtype=np.float32)
        out[:data.shape[0],:,:]=data.real
        out[data.shape[0]:,:,:]=data.imag
        return out
    def prepare_data(self,data):
        seq_frames,odd_frames=data[0],data[1]
        enc_targets=np.zeros((len(data[0]),1,2*data[0][0].shape[0],data[0][0].shape[1],data[0][0].shape[2]))
        dec_targets=np.zeros((len(data[1]),1,2*data[0][0].shape[0],data[0][0].shape[1],data[0][0].shape[2]))
        seq_frames_ip=np.zeros((len(data[0]),1,2*data[0][0].shape[0],data[0][0].shape[1],data[0][0].shape[2]))
        dec_ip=np.zeros((len(data[1]),1,data[1][0].shape[0],data[1][0].shape[1],data[1][0].shape[2]))
        for i,s in enumerate(seq_frames):
            seq_frames_ip[i,0,:,:,:]=self.convertcomplexto2d(s)
            if i<len(seq_frames)-1:
                enc_targets[i,0,:,:,:]=self.convertcomplexto2d(seq_frames[i+1])
            else:
                enc_targets[i,0,:,:,:]=self.convertcomplexto2d(seq_frames[i])
        for i,o in enumerate(odd_frames):
            dec_ip[i,0,:,:,:]=o
            dec_targets[i,0,:,:,:]=self.convertcomplexto2d(seq_frames[2*i+1])
        seq_frames_ip=torch.from_numpy(seq_frames_ip).cuda()
        dec_ip=torch.from_numpy(dec_ip).cuda()
        dec_targets=torch.from_numpy(dec_targets).cuda()
        enc_targets=torch.from_numpy(enc_targets).cuda()
        seq_frames_ip=seq_frames_ip.float().cuda()
        dec_ip=dec_ip.float().cuda()
        dec_targets=dec_targets.float().cuda()
        enc_targets=enc_targets.float().cuda()
        #print(enc_targets.type())
        return (seq_frames_ip,dec_ip,enc_targets,dec_targets)
    def __getitem__(self, idx):        
        cur_data=self.data[idx]
        frames=os.listdir(self.root_dir+'/'+cur_data)
        frames=frames[:40]
        frames.sort()
        buff=[]
        sequential_frames=[]
        odd_frames=[]
        for f in frames:
                img = cv2.imread(self.root_dir+'/'+cur_data+'/'+f)
                try:
                    imgfft=np.fft.fft(img)/800
                   # print(np.max(imgfft))
                    temp=np.fft.ifft(imgfft)
                    #print(temp)
                    temp=temp.astype('float64')
                    cv2.imwrite('5.jpg',temp)

                    imgfft=np.reshape(imgfft,(3,320,240))
                    buff.append(imgfft)
                except:
                    break
                
                
                
                
                if len(buff)==3:
                    b=np.zeros((4*buff[0].shape[0]+6*2,buff[0].shape[1],buff[0].shape[2]))    
                    sequential_frames.append(buff[0])
                    sequential_frames.append(buff[1])
                    temp=buff[2].copy()
                    b[:2*buff[0].shape[0],:,:]=self.convertcomplexto2d(buff[0])
                    b[2*buff[0].shape[0]+2*6:,:,:]=self.convertcomplexto2d(temp)
                    
                    odd_frames.append(b)
                    buff=[]
                    buff.append(temp)
        #print(len(sequential_frames))
        #cv2.destroyAllWindows()
        if len(odd_frames)==0:
            return None
        toreturn=self.prepare_data((sequential_frames,odd_frames))
        del sequential_frames
        del odd_frames
        return toreturn