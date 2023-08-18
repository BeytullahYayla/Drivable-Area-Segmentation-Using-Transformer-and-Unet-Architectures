import cv2
import numpy as np
import glob
import torch
frameSize = (1920, 1208)

out = cv2.VideoWriter('C:\\Users\\Beytullah\\Documents\\GitHub\\FordOtosan-L4Highway-Internship-Project\\data\\video\\output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)

for filename in glob.glob('C:\\Users\\Beytullah\\Desktop\\outputs\\*.jpg'):
    img = cv2.imread(filename)
    out.write(img)

out.release()