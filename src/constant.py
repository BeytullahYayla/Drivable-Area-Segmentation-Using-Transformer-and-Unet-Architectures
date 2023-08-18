import os

# Path to jsons
JSON_DIR = 'C:\\Users\\Beytullah\\Desktop\\Ford Otosan Staj Verileri\\ann'

# Path to mask
LINE_MASK_DIR  = 'C:\\Users\\Beytullah\\Desktop\\Ford Otosan Staj Verileri\\line_masks'
MASK_DIR  = 'C:\\Users\\Beytullah\\Desktop\\Ford Otosan Staj Verileri\\masks'
if not os.path.exists(LINE_MASK_DIR):
    os.mkdir(LINE_MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = 'C:\\Users\\Beytullah\\Desktop\\Ford Otosan Staj Verileri\\masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = 'C:\\Users\\Beytullah\\Desktop\\Ford Otosan Staj Verileri\\img_out'


# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = False

# Bacth size
BACTH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2

BACKGROUND=(0,0,0)
FREESPACE=(255,255,255)
RGB={
    
    0:BACKGROUND,
    1:FREESPACE
    
}