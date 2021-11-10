import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import csv
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

''' Read stored images
# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.015,0.025,0.015])
std = np.array([0.95,0.9,0.95])


class ImageDataset(Dataset):
    def __init__(self, root, image_shape):
        
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((img_height // 4, img_width // 4), Image.BICUBIC),
                transforms.ToTensor(),
                #print(transforms.ToTensor()),
                transforms.Normalize(mean, std),
            ]
        )
        
        print(root)
        self.files = sorted(glob.glob(root+'/*.png'))
        
        #print(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        #save_image(img,"Original image.png",normalize=False)
        #print(img)
        image = self.img_transform(img)
        #save_image(image,"LR image.png",normalize=False)
        


        return {"img": image}

    def __len__(self):
        return len(self.files)
'''

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img



def train_set():
    number_of_images = 1000
    level_of_noise = 3.5
    with open("/home/harshitha/Desktop/CVAsses/train_set.csv", 'w', newline='') as outFile:
        header = ['NAME', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(number_of_images):
            params, img = noisy_circle(200, 100, level_of_noise)
            np.save("/home/harshitha/Desktop/CVAsses/Data/" + str(i) + ".npy", img)
            #np.load("/home/harshitha/Desktop/CVAsses/Data" + str(i) + ".npy")
            write(outFile, ["/home/harshitha/Desktop/CVAsses/Data/" + str(i) + ".npy", params[0], params[1], params[2]])


def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])


if __name__ == '__main__':
    train_set()
