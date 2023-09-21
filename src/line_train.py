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
###
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
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score



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

def train_fn(id2label:dict,X_train,X_test,y_train,y_test,batch_size:int,):
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.size = 224
    feature_extractor.do_reduce_labels=False
    train_dataset=SemanticSegmentationDataset(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,train=True,id2label=id2label,feature_extractor=feature_extractor)
    valid_dataset=SemanticSegmentationDataset(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,train=False,id2label=id2label,feature_extractor=feature_extractor)
    label2id = {v:k for k,v in id2label.items()}
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,num_workers=0)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                         num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                         reshape_last_stage=True)
    
    optimizer = AdamW(model.parameters(), lr=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model Initialized!")
    print(device)
    for epoch in range(1, 6):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        #pbar = tqdm(train_dataloader)
        accuracies = []
        losses = []
        val_accuracies = []
        val_losses = []
        model.train()
        for idx, batch in enumerate(train_dataloader):
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(pixel_values=pixel_values, labels=labels)

            # evaluate
            upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            mask = (labels != 255) # we don't include the background class in the accuracy calculation
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()
            accuracy = accuracy_score(pred_labels, true_labels)
            loss = outputs.loss
            accuracies.append(accuracy)
            losses.append(loss.item())
            #pbar.set_postfix({'Batch': idx, 'Pixel-wise accuracy': sum(accuracies)/len(accuracies), 'Loss': sum(losses)/len(losses)})

            # backward + optimize
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(valid_dataloader):
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(pixel_values=pixel_values, labels=labels)
                    upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)

                    mask = (labels != 255) # we don't include the background class in the accuracy calculation
                    pred_labels = predicted[mask].detach().cpu().numpy()
                    true_labels = labels[mask].detach().cpu().numpy()
                    accuracy = accuracy_score(pred_labels, true_labels)
                    val_loss = outputs.loss
                    val_accuracies.append(accuracy)
                    val_losses.append(val_loss.item())

        print(f"Train Pixel-wise accuracy: {sum(accuracies)/len(accuracies)}\
            Train Loss: {sum(losses)/len(losses)}\
            Val Pixel-wise accuracy: {sum(val_accuracies)/len(val_accuracies)}\
            Val Loss: {sum(val_losses)/len(val_losses)}")
        
    torch.save(model,"best_line.pth")
    



if __name__=='__main__':

    id2label={
        0:"Background",
        1:"Solid  Line",
        2:"Dashed Line"
    }
    X_train,X_test,y_train,y_test= load_data()
    train_fn(id2label,X_train,X_test,y_train,y_test,batch_size=4)