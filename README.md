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
 - [Training](#training)
      *   [Splitting Data](#splitting_data)
      *   [Optimizer](#optimizer)
      *   [Loss Function](#loss-function)
      *   [Segmentation Model Selection](#segmentation-model-selection)
      *   [Training Loop](#training_loop)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
    
  

 
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

## Training
In this part we will train our model with our images and corresponding masks which we created by the help of json2mask script.
### Train-Validation-Test Split
Splitting data is crucial part of any data science related projects. We split our data into three group. Training, Validation and Test data.<br> <br>
<ul>
<li><b>Training Data:</b>Training data is used for train our model to learn specific patterns and adjusts it's parameters to understand the relationships between input data and corresponding outputs. The important topic about the training data is it shouldn't be used for evaluating model's performance.</li>
 <li><b>Validation Data:/b>During model training we use the <b>validation data</b> to asses it's performance and fine-tune hyperparameters. By evaluating our model on the validation data at every end of epoch we can prevent overfitting and gauge generalization problems.</li>
 <li><b>Test Data:</b>The test data is used to evaluate your model's performance in a real-world scenario. This data should not have been used during training or validation. A model's good performance on test data demonstrates its ability to generalize beyond the seen examples.</li>
</ul>



When i do quick search from the internet i found that generally 80:10:10 or 70:20:10 rate for bigger datasets (Training,Validation, Test) is used to train and evaluate machine learning models. When we look at the dataset we have, it consists of 4660 image which is not very small. So i decided to use 70:20:10 rate for evaluate my model's performance better. Here is the code that provides to split our training, validation and test data.
```
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * TEST_SIZE)
valid_ind = int(test_ind + len(indices) * VALID_SIZE)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//BATCH_SIZE
print(len(train_input_path_list))
```
Here is the constants to be used in training loop.
```
VALID_SIZE = 0.2
TEST_SIZE  = 0.1
BATCH_SIZE = 4
EPOCHS = 5
CUDA = True
INPUT_SHAPE = (224, 224)
N_CLASSES = 2

```

### Optimizer

In the context of machine learning and neural networks, refers to an algorithm or method used to adjust the parameters of a model in order to minimize the error or loss function. The goal of optimization is to find the optimal set of parameters that allow the model to best fit the training data and make accurate predictions on new, unseen data. Here are the examples of optimizer:

<ul>
 <li>
  Stochastic Gradient Descent:This is a basic optimization algorithm that updates the parameters in the opposite direction of the gradient of the loss function. It uses a small random subset (mini-batch) of the training data for each iteration to compute the gradient.
 </li>
 <li>
  Adam(Adaptive Moment Estimation):Adam combines the ideas of momentum and adaptive learning rates. It maintains exponential moving averages of past gradients and squared gradients to adjust the learning rate for each parameter individually. This helps the optimizer to adapt to different features and learning rates. Adam is  a popular algorithm because it achieves good results fast. Some of the hyperparameters of adam optimizer as follows:
  <ul>
   <li>
    alpha: Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training
   </li>
   <li>beta1: The exponential decay rate for the first moment estimates (e.g. 0.9).</li>
   <li>beta2: The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).</li>
   <li>epsilon: Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).</li>
  </ul>
Compared to other optimizer algorithms in mnist dataset results as follows. 
  
![adam](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/9269b01b-96ca-4110-bdea-19361d0b874a)

As we can see adam optimizer way better than other algorithms in terms of training cost.



 </li>
 <li>
  RMSProp: Similar to Adam, RMSprop also adapts the learning rate for each parameter. It divides the gradient by a moving average of the squared gradient, which helps in normalizing the gradient updates.
 </li>
 <li>Adagrad(Adaptive Gradient Algorithm):Adagrad (Adaptive Gradient Algorithm): Adagrad adapts the learning rate of each parameter based on the historical gradient information. It allocates larger updates to parameters with smaller historical gradients and smaller updates to parameters with larger historical gradients.</li>
</ul>

I selected adam optimizer between above optimizers because of it's popularity and generally being most efficient optimizer that combine momentum and rmsprob optimizers.

```
optimizer = AdamW(model.parameters(), lr=0.0001)

```

### Loss Funtion

As loss function i have used Binary Cross Entropy loss function. BCE a model metric that tracks incorrect labeling of the data class by a model, penalizing the model if deviations in probability occur into classifying the labels. It updates the parameters by propagating back to the network according to this loss value.

```
criterion = torch.nn.BCEWithLogitsLoss()

```
### Semantic Segmentation Model Selection
### Unet
When it comes to sementic segmentation tasks <b>U-NET</b> is one of the most popular model to achieve segmentation task. It's using convolutional neural networks to extract important features and updates image dimensions. Semantic segmentation, also known as pixel-based classification, is an important task in which we classify each pixel of an image as belonging to a particular class. U-net is a encoder-decoder type network architecture for image segmentation. U-net has proven to be very powerful segmentation tool in scenarios with limited data (less than 50 training samples in some cases). The ability of U-net to work with very little data and no specific requirement on input image size make it a strong candidate for image segmentation tasks.

![1_f7YOaE4TWubwaFF7Z1fzNw](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/6281f882-05c9-4c50-a9f2-a8622b70d899)

Unet consists of several part.
<ul>
 <li>
  Contracting Path:. It consists of convolutional layers, max-pooling layers, and sometimes batch normalization and activation functions like ReLU. This part of the network captures context and features from the input image.

 </li>
 <li>
  Bottleneck:At the bottom of the U shape is the bottleneck. This is typically a series of convolutional layers without pooling, aiming to capture detailed features and spatial information from the input.

 </li>
 <li>
  Expanding Path:The bottom part of the U shape is called the expanding path. It involves upsampling the features using techniques like transposed convolutions or bilinear interpolation. This part of the network aims to recover the spatial information lost during downsampling and provides high-resolution feature maps.
 </li>
 <li>
  Skip Connections:One of the key features of the U-Net architecture is the use of skip connections. During the expanding path, feature maps from the contracting path are concatenated with the upsampled feature maps. These skip connections help retain fine-grained details from the input image, which can improve the segmentation accuracy.
 </li>
 <li>
  Output Layer:The final layer of the network is a convolutional layer that produces the segmentation map. Depending on the task, it might use different activation functions and output channels.

 </li>
</ul>

### Pooling

Pooling is used to downsapling operations in forward propagation. . These layers play a crucial role in reducing the spatial dimensions of the feature maps while retaining essential information. We've used max pooling in our model.

```
self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
```

![8](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/b1a8dae4-7c83-4140-80ed-94167ac1cab9)

### Activation Function
When i look for a activation function after batch normalization layers, i found that ReLu activation function is very popular. With activation function we are introducing the property of non-linearity to a deep learning model and solving the vanishing gradients issue. The negative values default to zero, and the maximum for the positive number is taken into consideration. 

![image-10](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/c73ad4a2-b4de-4518-9b7c-0dcbb309d51a)

### Segformer

 SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perception (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding, thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from different layers, and thus combining both local attention and global attention to render powerful representations. We show that this simple and lightweight design is the key to efficient segmentation on Transformers.
 
![segformer_architecture](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/88d5f9c7-03d1-4c76-bd4b-680a6c12caa3)

### Evaluation

### Pixel-wise Accuracy
Pixel-wise accuracy is a metric used to evaluate the performance of image segmentation models, which assign a class label to each pixel in an image. It measures the percentage of correctly classified pixels in the predicted segmentation mask compared to the ground truth mask.

<i><b>Pixel-wise Accuracy</b>= (Total Number of Pixels/Number of Correctly Classified Pixels)*100</i>

Due to hardware constraints i used pretrained segformer model. Which is "nvidia/segformer-b0-finetuned-ade-512-512". This model pretrained before and gained good results. I retrained this model 5 epochs using Ford Otosan Highway data and afterall i got Train Pixel-wise accuracy: 0.9971257076378707         Train Loss: 0.007176180044638768         Val Pixel-wise accuracy: 0.9968600440612136         Val Loss: 0.007829783585266302 these accuracy values. As you can see our validation and training accuracies are close each other which means it seems there is no overfitting or underfitting which is great!
​
### IOU(Intersection Over Unit)

The Intersection over Union (IOU) score, also known as the Jaccard index, is a widely used evaluation metric in computer vision tasks, particularly in object detection and segmentation. It measures the overlap between two sets, typically used to assess the quality of predicted bounding boxes or segmentation masks compared to ground truth annotations.

In the context of segmentation tasks, the IOU score quantifies how well the predicted segmentation mask aligns with the ground truth mask. The formula for calculating the IOU score is:

<i>IOU=(Area of intersection)/(Area Of Union)</i>

```
def calculate_iou(ground_truth_mask,predicted_mask):
    
    intersection = np.logical_and(ground_truth_mask, predicted_mask)

    union = np.logical_or(ground_truth_mask, predicted_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    
    
```

After i created to this method, i calculated iou values of 100 test images and keep their scores in empty list. Then, i calculated a mean iou score to evaluate our model much better.

```
iou_scores=[]
for i in tqdm.tqdm(range(0, 100)):
    rand_idx=random.randint(0,len(X_test)-1)
    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
    image=plt.imread(X_test[i])
    mask=plt.imread(y_test[i])
    pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(device)
    model.eval()
    outputs = model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)
    logits = outputs.logits.cpu()
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
    overlay_img = img.copy()
    mask_alpha = 0.4
    img[color_seg[:,:,1]==255,:]=(255,0,255)
    iou_score=calculate_iou(mask,color_seg[:,:,1])
    iou_scores.append(iou_score)
    


mean_iou_score=sum(iou_scores)/100
```
The result that we get after this operations is 0.9894445434355832. 

### Inference

At the end of the training and evaluation operations lets we do some predictions. 

![ford_predictions](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/87e4676f-aee9-4128-bd81-90eb573c1ccd)



