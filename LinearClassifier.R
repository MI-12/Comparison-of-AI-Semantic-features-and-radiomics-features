
rm(list = ls())
gc()

library("glmnet")
library("pROC")
library("readxl")
library("nnet")


workbook<-"Feature matrix extracted by BigbiGAN from yout training dataset."
Sta = read_excel(workbook,1)

workbook1<-"Feature matrix extracted by BigbiGAN from yout test dataset."
Sta1 = read_excel(workbook1,1)

workbook2<-"Feature matrix extracted by BigbiGAN from yout validation dataset."
Sta2 = read_excel(workbook2,1)

workbook3<-"Feature matrix extracted by BigbiGAN from yout external validation dataset."
Sta3 = read_excel(workbook3,1)

Train_Ori<-data.frame(Sta)
Tval<-data.frame(Sta1)
Test<-data.frame(Sta2)
ExterV<-data.frame(Sta3)


FeaNumTrain<-"Dimension of semantic features extracted by BigBiGAN"
FeaNumTest<-"Dimension of semantic features extracted by BigBiGAN"
FeaNumTval<-"Dimension of semantic features extracted by BigBiGAN"
FeaNumExtV<-"Dimension of semantic features extracted by BigBiGAN"



TrainFea<-data.frame(Train)
TvalFea<-data.frame(Tval)
TestFea<-data.frame(Test)
ExterVFea<-data.frame(ExterV)



Trainy<-Train[(ncol(Train)-FeaNumTrain):(ncol(Train))] ##Features are stored at the end of the matrix.
Trainy<-data.frame(Trainy)

fm <- lm("COVID"~., data=Trainy)
summary(fm)

lmpred_Val<-predict(fm,Tval)
Roc_Val<-roc(Tval$"COVID",lmpred_Val)
auc(Roc_Val)
plot(Roc_Val, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

lmpred_Test<-predict(fm,Test)
Roc_Test<-roc(Test$"COVID",lmpred_Test)
auc(Roc_Test)
plot(Roc_Test, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

lmpred_Ext<-predict(fm,ExterV1)
Roc_Ext<-roc(ExterV1$"COVID",lmpred_Ext)
auc(Roc_Ext)
plot(Roc_Ext, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)