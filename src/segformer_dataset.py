import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self,X_train,X_test,y_train,y_test,id2label,feature_extractor,train=True,transforms=None):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        #self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.train=train
        self.transforms=transforms
        

        #self.classes_csv_file = os.path.join(self.root_dir, "_classes.csv")
        
        self.id2label =id2label #{x[0]:x[1] for x in data}
        
        """"
        image_file_names = [f for f in os.listdir(self.root_dir) if '.jpg' in f]
        mask_file_names = [f for f in os.listdir(self.root_dir) if '.png' in f]
        
        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)
        """

    def __len__(self):
        if self.train:
            
            return len(self.X_train)
        else:
            return len(self.X_test)

    def __getitem__(self, idx):
        
       
            
        if self.train:
            
            image = plt.imread(self.X_train[idx])
            #Image.open(os.path.join(self.root_dir, self.images[idx]))
            segmentation_map = plt.imread(self.y_train[idx])
        else:
            image=plt.imread(self.X_test[idx])
            segmentation_map=plt.imread(self.y_test[idx])
            
        

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs