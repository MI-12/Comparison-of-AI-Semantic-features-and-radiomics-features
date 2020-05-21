# Unet_and_BigBiGAN_for_COVID-19
## The source code of Unet for pneumonia lesion segmentation, and BigBiGAN training for semantic feature extreaction.

### In order to achieve reproducibility, please execute the following code in the order mentioned in the paper: Unet-BibBiGAN-Linear Classifier.

* Detailed instruction of using the code is as follows.

* ModelUnet: The main function of the improved Unet for pneumonia lesion segmentation.

* HDF5_Read: Read the images when using ModelUnet for segmentation.

* Section_Unet and Section_Unet_parts: Functions for Unet.

* Unet_Train_GIoU: GIoU loss for segmentation.

* UnetTrain_Gen: The details of the Unet code execution.

* VGG19: VGG feature extraction.

* Weight_Init: Initialization of the network.

* BigBiGAN_Learning: Training of the BigBiGAN for sementic feature extraction.

* LinearClassifier.R: Classification based on the features extracted by the BigBiGAN.
