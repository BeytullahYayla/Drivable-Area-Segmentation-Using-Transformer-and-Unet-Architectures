import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import torchvision
import torchvision.transforms.functional as TF
import random
import glob
from torchmetrics import JaccardIndex
from torchmetrics.classification import Dice
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../src/')
import preprocess
#import json2mask
import constant
from model import Unet
from segformer_dataset import SemanticSegmentationDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import random
from collections import OrderedDict
import pickle

def load_data():
    ###

    # X_train'i yükleme
    with open('../X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)

    # X_test'i yükleme
    with open('../X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)

    # y_train'i yükleme
    with open('../y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    # y_test'i yükleme
    with open('../y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    return X_train,X_test,y_train,y_test

def predict(image,mask,RGB:dict,feature_extractor_inference,model):
    image=plt.imread(image)
    mask=plt.imread(mask)
    
    model.eval()
    pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to("cuda")

    outputs = model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)
    logits = outputs.logits.cpu()
    
    model.eval()
    outputs = model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)
    logits = outputs.logits.cpu()
    fig,axs=plt.subplots(ncols=4,nrows=1,figSize=(5,5))
        # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(logits,
                    size=image.shape[:-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)

    # Second, apply argmax on the class dimension
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3\
    for label, color in enumerate(RGB):
        color_seg[seg == label, :] = RGB[label]
    # Convert to BGR
        color_seg = color_seg[..., ::-1]

    # Show image + mask
        img = np.array(image)


        axs[0][0].set_title('Original Image')
        axs[0][1].set_title('Ground Truth')
        axs[0][2].set_title("Predicted Mask")
        axs[0][3].set_title('Predicted Mask On Image')
        axs[0][0].imshow(img)
        axs[0][1].imshow(mask)
        axs[0][2].imshow(color_seg[:,:,1])
        overlay_img = img.copy()
        mask_alpha = 0.4
        img[color_seg[:,:,1]==255,:]=(255,0,255)

        axs[0][3].imshow(img)


if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    SOLID_LINE=(255,0,0)
    BACKGROUND=(255,255,255)
    DASHED_LINE=(0,255,0)
    RGB={
    
    0:BACKGROUND,
    1:SOLID_LINE,
    2:DASHED_LINE
    
    
}
    id2label={
        0:"Background",
        1:"Solid  Line",
        2:"Dashed Line"
    }
    feature_extractor_inference=SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
    image_to_predict=X_train[0]
    mask=y_train[0]
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.size = 224
    feature_extractor.do_reduce_labels=False
    train_dataset=SemanticSegmentationDataset(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,train=True,id2label=id2label,feature_extractor=feature_extractor)
    valid_dataset=SemanticSegmentationDataset(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,train=False,id2label=id2label,feature_extractor=feature_extractor)
    label2id = {v:k for k,v in id2label.items()}
    model=torch.load("C:\\Users\\Beytullah\\Desktop\\models\\best_line.pth")
    predict(image_to_predict,mask,RGB,feature_extractor_inference,model)

