
import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

import CircleDetectionModel
from CircleDetectionModel import *

from Dataset import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import functools
from torch.distributions import normal
import torchvision.datasets as dset

import csv

from sklearn import preprocessing
import time 
import pandas as pd

#from apex.parallel import DistributedDataParallel as DSP


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--image_size", type=int, default=200, help="image size of noisy circle")
parser.add_argument("--img_radius", type=int, default=100, help="radius of the circle")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--dim_z", type=int, default=64, help="Latent space input size[noise input]")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--n_residual_blocks',type=int,default=8,help='num of residual block layers')
parser.add_argument('--train',type=str,default='Circle',help='Detect shapes')
parser.add_argument('--n_gpus',type=int,default=1,help='number of GPUs')
opt = parser.parse_args()
print(opt)

dataset_name=opt.dataset_name

os.makedirs("Images_%s_%d" %(dataset_name,opt.n_epochs), exist_ok=True)
os.makedirs("saved_models_%s_%d" %(dataset_name,opt.n_epochs), exist_ok=True)
os.makedirs("Loss_%s_%d" %(dataset_name,opt.n_epochs), exist_ok=True)


lr_init=opt.lr
lr_decay=0.1
decay_every=opt.n_epochs


cuda = torch.cuda.is_available()
device=torch.device('cuda')


img_shape = (opt.channels, opt.img_size, opt.img_size)
gpu=opt.n_gpus

z=normal.Normal(0,1.0)
z=z.sample([opt.batch_size, opt.dim_z])
#print("Noise",z.shape)



detectcircle=Detection(batch_size=opt.batch_size)

class NoisyImages(Dataset):
    def __init__(self, path_to_trainset, transform=None):
        self.dataset = pd.read_csv(opt.dataroot, sep=' ')
        self.transform = transform

    def __getitem__(self, idx):
        image = np.load(self.dataset.iloc[idx, 0].split(',')[0])
        target = [self.dataset.iloc[idx, 0].split(',')[i] for i in range(1, 4)]
        
        if self.transform:
            image = np.expand_dims(np.asarray(image), axis=0)
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            target = torch.from_numpy(np.array(np.asarray(target), dtype=np.float32))
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)


# Losses
criterion_conv = torch.nn.BCEWithLogitsLoss()
criterion_loss = torch.nn.L1Loss()
criterion_percep=torch.nn.MSELoss()

if cuda:
    
    z=z.cuda()
    detectcircle=detectcircle.cuda()
    criterion_percep=criterion_percep.cuda()
    criterion_loss=criterion_loss.cuda()
    criterion_conv=criterion_conv.cuda()


if opt.epoch != 0:
    # Load pretrained models
    
    detectcircle=torch.load("/home/harshitha/Desktop/CVAsses/saved_models/detectcircle_%d.pth"%(opt.epoch))

# Optimizers

optimizer_detectcircle=torch.optim.Adam(detectcircle.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
print(opt.dataset)

normalize = transforms.Normalize(mean=[0.5], std=[0.5])
if opt.dataset=='CVDetect':
  trainset = NoisyImages(
        opt.dataroot,
        transforms.Compose([
            normalize,
        ]))
  print(len(opt.dataset_name))
  dataloader = DataLoader(dataset=trainset,
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=opt.n_cpu)




# ----------
#  Training
# ----------

def train():
  
    if torch.cuda.device_count() == 1:
        print("Only one GPU",torch.cuda.device_count())
    start_overall_time=time.time()
    n_epochs=str(opt.n_epochs)
    which_dataset_name=str(opt.dataset)
    path=r"Loss_%s_%d" %(which_dataset_name,opt.n_epochs)
    path=path+'CicleDetect_loss_epoch'+n_epochs+which_dataset_name+'.csv'
    
    with open(path,'w') as loss:
                
        loss_file = csv.writer(loss)
        loss_file.writerow(["Epoch","Total epochs","Iter","CircleDetection Loss"])
    
    for epoch in range(opt.epoch, opt.n_epochs):
        start_time=time.time()
        circle_train_loss1=0.0
        
        for i, (imgs,target) in enumerate(dataloader):
  
          # Configure model input
          imgs = imgs.type(Tensor)
          imgs=imgs.to(device)
          
          print(imgs.shape)
          
          target=target.to(device)
          
          #Train detectcircle

          optimizer_detectcircle.zero_grad()
          

          circle_detect_res=detectcircle(imgs)
          
          print('circle detection',circle_detect_res, 'target', target)
          
 
          circledetection_loss=criterion_loss(circle_detect_res,target/200)*0.1+criterion_percep(circle_detect_res,target/200)

          
          circledetection_loss.backward()

          optimizer_detectcircle.step()

          circle_train_loss1+=circledetection_loss.item()

                    

          # --------------
          #  Log Progress
          # --------------

          sys.stdout.write(
              "[Epoch %d/%d] [Batch %d/%d] [Detect loss: %f] \n"
              % (epoch, opt.n_epochs, i, len(dataloader), circledetection_loss.item())
          )
          n_epochs=str(opt.n_epochs)
          which_dataset_name=str(opt.dataset)
          path=r"Loss_%s_%d" %(which_dataset_name,opt.n_epochs)
          path=path+'CicleDetect_loss_epoch'+n_epochs+which_dataset_name+'.csv'
          
            
          with open(path,'a') as loss:
              loss_file = csv.writer(loss, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
              loss_file.writerow([epoch, opt.n_epochs, i, circledetection_loss.item()])
          
          batches_done = epoch * len(dataloader) + i

          if batches_done % opt.sample_interval == 0:
           
            save_image(imgs,"Images_%s_%d/%d.jpeg" %(dataset_name,opt.n_epochs,batches_done),normalize=True)
            
        end_time=time.time()
        total_time=end_time-start_time
        print(epoch,opt.checkpoint_interval,total_time)
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
          
          torch.save(detectcircle, "saved_models_%s_%d/circledetection_%d.pth" %(dataset_name,opt.n_epochs,epoch))
    
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for CircleDetect)" % (lr_init * new_lr_decay)
            print(log)
    end_overall_time=time.time()
    overall_time=end_overall_time-start_overall_time
   

    sys.stdout.write("[Epoch %d]  [Time to complete all epochs %f] [Detection loss %f]\n"
                     %(opt.n_epochs,overall_time, circle_train_loss1))



if opt.train=='Circle':
    train()
