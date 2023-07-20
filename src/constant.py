import os

# Path to jsons
JSON_DIR = 'C:\\Users\\Beytullah\\Documents\\GitHub\\FordOtosan-L4Highway-Internship-Project\\data\\jsons'

# Path to mask
LINE_MASK_DIR  = '../data/line_masks'
MASK_DIR  = '../data/masks'
if not os.path.exists(LINE_MASK_DIR):
    os.mkdir(LINE_MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = '../data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = '../data/images'


# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = True

# Bacth size
BACTH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2