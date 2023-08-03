# FordOtosan-L4Highway-Internship-Project
# 🚩 Contents
- [Project Structure](#-project-structure)
- [Topics](#-pages)
  * [General Overview to Project Topics](https://sway.office.com/GWVpEgbbvFCstcKv?ref=Link)
- [Getting Started](#Getting-Started)
  *  [Json2Mask](#json2mask)
  *  [Mask On Image](#mask-on-image)
  *  [Preprocessing](#preprocessing)
      *   [torchlike_data](#torchlike_data-method)
      *   [one_hot_encoder](#one_hot_encoder-method)
      *   [tensorize_image](#tensorize_image-method)
      *   [tensorize_mask](#tensorize_mask-method)

 
# 🗃 Project Structure
```
Ford Otosan Level 4 Higway Autonomus Vehicle Freespace Segmentation Project

├─ README.md
├─ requirements.txt
├─ src
|   ├─ constant.py
|   ├─ deeplabv3.py
|   ├─ json2line.py
|   ├─ json2mask.py
|   ├─ mask_on_image.py
|   ├─ model.py
|   ├─ preprocess.py
|   └─ train.py
├─ notebooks
|   ├─ overview.ipynb
|   └─ .ipynb_checkpoints
|       ├─ data_overview-checkpoint.ipynb
|       └─ overview-checkpoint.ipynb
└─ data
    └─ images
    |    ├─ cfcu_002387.png
    └─ jsons
    |    ├─ cfcu_002387.png.json
    └─ line_masks
    |    ├─ cfcu_002387.png
    └─ masked_images
    |    ├─ cfcu_002387.png
    └─ masks
         └─ cfcu_002387.png
```

# Getting Started
## The purpose of the project
In this project my aim is to detect drivable areas in highways using state of the art deep learning techniques and present a repository to autonomous vehicle engineers.

## Json2Mask
This script file enables us to create masks using provided json files. Every json file has similar structre as follows.
```
        {
            "classId": 38,
            "classTitle": "Freespace",
            "createdAt": "2020-06-25T08:31:34.965Z",
            "description": "",
            "geometryType": "polygon",
            "id": 89201,
            "labelerLogin": "odogan13",
            "points": {
                "exterior": [
                    [
                        1920,
                        1208
                    ],
                    [
                        1920,
                        605
                    ],
                    [
                        1684,
                        604
                    ],
                    [
                        1670,
                        598
                    ],
                    [
                        1600,
                        586
                    ],
                    ...
                ],
                "interior": []
            },
            "tags": [],
            "updatedAt": "2020-07-10T10:37:41.236Z"
        },
```
Let we dive into deeper about this json file. The important features of this json are <b>classTitle</b>(to determine which object properties we are seeing or basically the class name of the object), <b>points,exterior</b>(We're gonna create mask by this specific points). <b>Exterior</b> refers to list of coordinates of the points that make up the outer edge of the polygon. You can take a look example mask that i obtained in jupyter notebook using json2mask script. Lets examine json2mask.py file together.

```
json_list = os.listdir(JSON_DIR)

iterator_example = range(1000000)

for i in tqdm.tqdm(iterator_example):
    pass



# For every json file
for json_name in tqdm.tqdm(json_list):

    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')

    # Load json data
    json_dict = json.load(json_file)

    # Create an empty mask whose size is the same as the original image's size
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    mask_path = os.path.join(MASK_DIR, json_name[:-9]+".png")

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle']=='Freespace':
            # Extract exterior points which is a point list that contains
            # every edge of polygon and fill the mask with the array.
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)

    # Write mask image into MASK_DIR folder
    cv2.imwrite(mask_path, mask.astype(np.uint8))

```

From top to bottom we basically get the name eof every json file in json_list and we join with Json directory to open and access provided json files. Then, we convert json file to json dict using json library. To draw mask first we need to know about size of images. We get this information from json dictionary and creating empty canvas to annotate corresponding pixels as freespace. After that we check all objects part of json dictionary if it is freespace then we can draw it. Lastly we save mask that we created using <b>cv2.imwrite</b> method.


![Json2Mask](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/664a03e3-254c-49f9-acb9-5cda517f6340)
![Json2MaskLine](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/e40e3008-0bd5-4b39-9404-4fcde68e24ba)


## Mask on Image
```
# For every mask image
for mask_name in tqdm.tqdm(mask_list):
    # Name without extension
    mask_name_without_ex = mask_name.split('.')[0]

    # Access required folders
    mask_path      = os.path.join(MASK_DIR, mask_name)
    image_path     = os.path.join(IMAGE_DIR, mask_name_without_ex+'.png')
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)

    # Read mask and corresponding original image
    mask  = cv2.imread(mask_path, 0).astype(np.uint8)
    image = cv2.imread(image_path).astype(np.uint8)

    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image
    cpy_image  = image.copy()
    image[mask==1, :] = (255, 0, 125)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)

    # Write output image into IMAGE_OUT_DIR folder
    cv2.imwrite(image_out_path, opac_image)
```
First, in every mask in mask_list we obtain mask_name and for every mask name we remove it's extensions and create mask_path, image_path, image_out_path which corresponds where to save result images. After that we read mask as grayscale format and images. Then change the color of pixels on the original image that corresponds to the mask part and create new image. Finally we write output image into <b>IMAGE_OUT_DIR</b> folder.

![image](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/fe876bac-78bb-4e74-81b6-6e346df6e04d)

## Preprocessing
In image classification and segmentation task we generally use preprocessing techniques. Preprocessing techniques refer to a set of data preparation steps that are applied to raw data before it can be used for analysis or modeling. Preprocessing aims to clean, organize and transform data so that it becomes suitable for further processing. In our project we have preprocess script. This script provides us to tensorize, resize and encode input data to give segmentation model properly. I'm gonna introduce you to methods one by one.

### torchlike_data() method
This function is intended to convert the given input array data into a structure similar to a PyTorch tensor. The input array has a shape of "HxWxC", where H represents the height, W denotes the width, and C represents the number of channels. For example, in the case of a color image, C=3 for the RGB channels.
The converted output data will have the shape "CxHxW", where each channel will be a separate layer or matrix. This means that each channel will be treated as an individual 2D matrix containing data corresponding to the original input array's respective channel.

```
 def torchlike_data(data):


    # Obtain channel value of the input
    n_channels = data.shape[2]

    # Create and empty image whose dimension is similar to input
    torchlike_data_output = np.empty((n_channels,data.shape[0],data.shape[1]))

    # For each channel
    for i in range(n_channels):
        torchlike_data_output[i]=data[:,:,i]

        

    return torchlike_data_output
```
The logic of method is as follows:<br>
<ol>
 <li>
 The number of channels n_channels is obtained from the third dimension (channels) of the input array.
  
 </li>
 <li>
 An empty output array torchlike_data_output is created. It will have dimensions C (number of channels), H (height), and W (width) to store each channel's data as a separate matrix.

  
 </li>
 <li>
  The function uses a for loop to iterate over each channel.
 </li>
 <li>
  The data for each channel is copied into the torchlike_data_output array by assigning the corresponding channel data from the data array (data[:,:,i]).

 </li>
 <li>
  As a result, the data for each channel will be stored as a separate matrix within the torchlike_data_output array. This data organization mimics the structure of PyTorch tensors.

 </li>
</ol>

### one_hot_encoder() method
Encoding categorical data is a crucial part of most of the data science projects. In this approach, each category is represented as a binary vector with a length equal to the number of unique categories. The vector has a value of 1 in the position corresponding to the category and 0 in all other positions. One-hot encoding is preferred when there is no natural order between categories, and it avoids introducing any ordinal relationship between them.
```
def one_hot_encoder(data, n_class):

    if len(data.shape) != 2:
        print("It should be same with the layer dimension, in this case it is 2")
        return
    if len(np.unique(data)) != n_class:
        print("The number of unique values ​​in 'data' must be equal to the n_class")
        return

    # Define array whose dimensison is (width, height, number_of_class)
    encoded_data = np.zeros((*data.shape, n_class), dtype=np.int)

    # Define labels
    if n_class==2:
        encoded_labels = [[0,1], [1,0]]#Freespace or not
    if n_class==3:
        encoded_labels=encoded_labels = [
        [1, 0, 0],  # No line
        [0, 1, 0],  # Solid Line
        [0, 0, 1],  # Dashed Line
    ]

    
    for i,unique_val in enumerate(np.unique(data)):
        encoded_data[data==unique_val]=encoded_labels[i]



    return encoded_data
```
The logic of method is as follows:<br>
<ol>
 <li>
 First we check shape of input data. It's shape must be equal to two and number of unique pixels must be equal to number of classes. Otherwise method returns nothing and finished to execution.
 </li>
 <li>
 Secondly, an empty output array encoded_data is created. It will have same dimensions with data. As a third channel we give number of classes.
 </li>
 <li>
  We create encoded_labels vector by number of classes we have.
 </li>
 <li>
  In for loop assigns to each element in the data array whose value is unique_val, the value at the corresponding index in another array called encoded_labels. That is, the encoded_labels array represents the encoded version of the data array.
 </li>

</ol>

### tensorize_image() method
This method combines of our above preprocessing methods and returns final tensor output to give our segmentation model.
```
def tensorize_image(image_path_list, output_shape, cuda=False)

    # Create empty list
    local_image_list = []

    # For each image
    for image_path in image_path_list:

        # Access and read image
        image = cv2.imread(image_path)

        # Resize the image according to defined shape
        image = cv2.resize(image, output_shape)

        # # Normalize image
        # image=image//255

        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(image)

        # Add into the list
        local_image_list.append(torchlike_image)

    # Convert from list structure to torch tensor
    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()

    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()

    return torch_image
```

The logic of method is as follows:<br>
<ol>
 <li>
 First we create an empty list that keeps our image paths. 
 </li>
 <li>
 After that with for loop we read images, resize, normalize it  and transpoze channel placements with torchlike_data method
 </li>
<li>Finally we convert local_image_list into tensor from list data type and return it</li>


</ol>

### tensorize_mask() method

This method combines of our above preprocessing methods and returns final tensor output to give our segmentation model. In addition to tensorize_image method we encode our mask by calling one_hot_encoding method.

```
def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):
    
    # Create empty list
    local_mask_list = []

    # For each masks
    for mask_path in mask_path_list:

        # Access and read mask
        mask = cv2.imread(mask_path, 0)

        # Resize the image according to defined shape
        mask = cv2.resize(mask, output_shape)

        # Apply One-Hot Encoding to image
        mask = one_hot_encoder(mask, n_class)

        # Change input structure according to pytorch input structure
        torchlike_mask = torchlike_data(mask)


        local_mask_list.append(torchlike_mask)

    mask_array = np.array(local_mask_list, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()
    if cuda:
        torch_mask = torch_mask.cuda()

    return torch_mask

```

The logic of method is as follows:<br>
<ol>
 <li>
 First we create an empty list that keeps our mask paths. 
 </li>
 <li>
 After that with for loop we read , resize, normalize, encode it  and transpoze channel placements with torchlike_data method
 </li>
<li>Finally we convert local_mask_list into tensor from list data type and return it</li>


</ol>





