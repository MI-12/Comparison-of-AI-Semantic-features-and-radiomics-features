import numpy as np
import argparse
import os
import cv2
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import models 
import torch.optim as optim

from datetime import datetime
from HDF5_Read import *
import sys
from VGG19 import *
from Section_Unet import *
from Weight_Init import *
from UnetTrain_Gen import *
from Unet_Train_GIoU import *

class Unet:

    def __init__(self, args):

        # parameters
        self.lr = args.learning_rate
        self.epoch = args.epoch
        self.d_iter = args.d_iter
        self.batch_size = args.batch_size
        self.dicom_Height = args.dicom_Height
        self.dicom_Width = args.dicom_Width
        self.dataset = []
        self.dataloader = []

        self.lambda_gp = args.lambda_gp
        self.lambda_mse = args.lambda_mse
        self.lambda_vgg = args.lambda_vgg
        self.lambda_giou = args.lambda_giou

        self.gpu = False

        self.generator = Section_Unet().to("cuda:0") 
        self.Train_GIoU = Unet_Train_GIoU().to("cuda:0")
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))

        if torch.cuda.is_available():
            if ',' in args.gpu_ids:
                gpu_ids = [int(ids) for ids in args.gpu_ids.split(",")]
                print(gpu_ids)
            else:
                gpu_ids = int(args.gpu_ids)

            if type(gpu_ids) is not int:

                self.vgg19 = nn.DataParallel(self.vgg19, device_ids = gpu_ids)
                self.generator = nn.DataParallel(self.generator, device_ids=gpu_ids)
                self.Train_GIoU = nn.DataParallel(self.Train_GIoU, device_ids=gpu_ids)
            self.gpu = True
    
        if not self.load_model():
            initialize_weights(self.generator)

    def train(self, args):

        self.save_parameters()
        self.Traindataset = H5Dataset(args.input_dir_train)
        self.Traindataloader = torch.utils.data.DataLoader(
        self.Traindataset,
        batch_size=self.batch_size,
        num_workers=4, 
        shuffle=False,
        drop_last = True)

        for batch_index in range(0, self.epoch):
           
            self.generator.train()

            for i, (Ori_img, Seg_img, Name_img) in enumerate(self.Traindataloader):

                Ori_img = Ori_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                Seg_img = Seg_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
            
                loss_g = UnetTrain_Gen(self, Seg_img, Ori_img, batch_index)

            if ((batch_index + 1) % 4 == 0 and self.lr > 1e-7):
                self.G_optimizer.defaults["lr"] *= 0.5
                self.lr *= 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_train', type=str, default="Your Training Dataset")
    parser.add_argument('--gpu_ids', type=str, default="Your GPUs")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default="Your Batch_size")
    parser.add_argument('--epoch', type=int, default="Your Epoch")
    parser.add_argument('--dicom_Height',type=int, default="Default: 512")
    parser.add_argument('--dicom_Width',type=int, default="Default: 512")
    parser.add_argument('--lambda_mse', type = float, default = "Default: 1.0")
    parser.add_argument('--lambda_vgg', type = float, default = "Default: 1e-1")
    parser.add_argument('--lambda_giou', type = float, default = "Default: 1.0")

    args = parser.parse_args()
    munet = Unet(args)
    munet.train(args)
