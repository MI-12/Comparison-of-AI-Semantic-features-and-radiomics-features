import os
import pydicom
import SimpleITK as sitk
import six
from glob import glob
import numpy as np
from radiomics import featureextractor, getTestCase

datadir = 'Your Datasets'
Data_dataset = 'Your Datasets'

list_datasets = os.listdir(Data_dataset)
for dir_P_N_ in list_datasets:
    list_P_N_ = os.listdir(dir_P_N_)
    for list_dirs_Slices in list_P_N_:
        for dirs_Pat_in_Slices in range(len(list_dirs_Slices)):
            data_train_Ori_Pat = (os.path.join(Data_dataset,dirs_Pat_in_Slices[Ori]) #The original CT image
            data_train_Seg_Pat = (os.path.join(Data_dataset,dirs_Pat_in_Slices[Seg]) #The corresponding manual label

            imageName, maskName = data_train_Ori_Pat, data_train_Seg_Pat
            params = os.path.join(datadir, "examples", "exampleSettings", "Params.yaml")

            extractor = featureextractor.RadiomicsFeatureExtractor(params)
            result = extractor.execute(imageName, maskName)

            for key, val in six.iteritems(result):
                #print("\t%s: %s" %(key, val))
                with open("./Features.file", "a+") as fGIoU:
                    fGIoU.write(str(key) + ' ' + str(val)+ ' ')
                    
            with open("./Features.file", "a+") as fGIoU:
                fGIoU.write(os.linesep)
