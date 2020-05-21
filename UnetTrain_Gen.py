import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import cv2
import skimage

def UnetTrain_Gen(self, free_img, noised_img, batch_index, train=True):
    z = Variable(noised_img)
    real_img = Variable(free_img, requires_grad=False) 

    if self.gpu:
        z = z.cuda()
        real_img = real_img.cuda()

    self.G_optimizer.zero_grad() 

    criterion_mse = nn.MSELoss()
    criterion_vgg= nn.MSELoss()
    GIoU_loss = torch.tensor(0).cuda()
    z_sque = z.squeeze(1)
    fake_img = self.generator(z_sque).cuda()
    fake_img = fake_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                    
    mse_loss = criterion_mse(fake_img, real_img)

    fake_img_samples  =  fake_img

    feature_fake_vgg = self.vgg19(fake_img)
    feature_real_vgg = Variable(self.vgg19(real_img).data, requires_grad=False).cuda()
    vgg_loss = criterion_vgg(feature_fake_vgg, feature_real_vgg)

    giou_batch = torch.zeros(1, fake_img.shape[0])
    Threshold_batch = torch.zeros(1, fake_img.shape[0])
    GIoU_loss_batch = torch.zeros(1, fake_img.shape[0])
    Dice_batch = torch.zeros(1, fake_img.shape[0])
    NumberPoint = torch.zeros(1, fake_img.shape[0])

    if(batch_index > 100):
        NumberPoint, GIoU_loss, giou_batch, Threshold_batch,Dice_batch = Unet_Train_GIoU(fake_img, real_img)
        GIoU_loss_batch = GIoU_loss_batch.view(1,-1)
        giou_batch = giou_batch.view(1,-1)
        Threshold_batch = Threshold_batch.view(1,-1)
        Dice_batch = Dice_batch.view(1,-1)
        GIoU_loss = GIoU_loss_batch.mean()

        g_loss = self.lambda_giou * GIoU_loss + self.lambda_vgg * vgg_loss
    
    g_loss =  self.lambda_mse * mse_loss + self.lambda_vgg * vgg_loss
    
    if train:

        g_loss.backward()
        self.G_optimizer.step()

    return g_loss.data.item(), self.lambda_giou * GIoU_loss.data.item(),\
         self.lambda_mse * mse_loss, mse_loss, \
             fake_img_samples, giou_batch, \
                 Threshold_batch,Dice_batch,NumberPoint

