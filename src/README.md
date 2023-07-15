# Freespace Segmentation with Fully Convolutional Neural Network (FCNN)

## Build

1. In order to clone the repo to your local computer:

    ```bash
     git clone https://github.com/cakirogluozan/fo-intern-project
    ```

    or directly download from the button (but it is not recommended.)

2. (Optional) It is suggested to use virtual environment, in the root directory:
    - create a virtual environment:

        ```bash
        python3 -m venv venv
        ```

    - activate the virtual environment:

        ```bash
        source venv/bin/activate
        ```

    - deactivate the virtual environment:

        ```bash
        deactivate
        ```

    > YOU HAVE TO BE IN THE ROOT DIRECTORY FOR THIS COMMANDS.

3. Install required libraries:

    ```bash
     pip install -r requirements.txt
    ```

# Project

[Click on this link](https://youtu.be/i_rToxP3Txo) for a video that briefly explains the difference between machine learning and deep learning. After watching this video please search and answer the following questions:

- What is Machine Learning?
- What is Unsupervised vs Supervised learning difference?
- What is Deep Learning?
- What is Neural Network (NN)?
- What is Convolution Neural Network (CNN)?  Please give 2 advantages over NN.
- What is segmentation task in NN? Is it supervised or unsupervised?
- What is classification task in NN? Is it supervised or unsupervised?
- Compare segmentation and classification in NN.
- What is data and dataset difference?
- What is the difference between supervised and unsupervised learning in terms of dataset?

## Data Preprocessing

### Extracting Masks

- What is color space ?
- What RGB stands for ?
- In Python, can we transform from one color space to another?
- What is the popular library for image processing?

In this part of the project, we want you to convert every JSON file into mask images:

1. Move json files into data/jsons folder.
2. Open the src folder and run `json2mask.py` There is a video explaining the code: [Please click for English Lecture](https://youtu.be/p_JnbbSAxmU). [Please click for Turkish Lecture](https://youtu.be/P8OJ2JTiJa4).

    For those who want to follow the steps one by one:

    1. You need to move your files into data/jsons folder.
    2. You need to create a list which contains every file name in jsons folder.
    3. In a for loop, you need to read every json file and convert them into json dictionaries.
    4. You need to get width and height of image.
    5. You need to create an empty mask which will be filled with freespace polygons.
    6. You need to get objects in the dictionary, and in a for loop, you need to check the objects 'classTitle' are 'Freespace' or not.
    7. If it is a Freespace object, then you need to extract 'points' then 'exterior' of points which is a point list that contains every edge of polygon you clicked while labeling.
    8. You need to fill the mask with the array.
    9. You need to write mask image into data/masks folder. 

    ```bash
    cd src/
    python3 json2mask.py
    ```

3. To check mask files, run `mask_on_image.py` There is a video explaining the code: [Please click for Turkish Lecture.](https://youtu.be/xBA72K2Bp5E)

    ```bash
    python3 mask_on_image.py
    ```

- What do you think these masks will be used for? (Feature ? Label ?)

### Converting into Tensor

Before go on, please search and answer following questions:

- Explain Computational Graph.
- What is Tensor?
- What is one hot encoding?
- What is CUDA programming? Answer without detail.

The images and masks refer to "features" and "labels" for Segmentation. To feed them into the Segmentation model, which will be written in PyTorch, we need to format them appropriately. In this part, we will solve this issue. In the `preprocess.py` There is a video explaining the code: [Please click for Turkish Lecture.](https://youtu.be/znP2-rSK_QE)  There are two helper functions:

1. To convert images to tensor, we need  `tensorize_mask(.)`. For this complete `torchlike_data(.)`
2. To convert masks to tensor, we need `tensorize_mask(.)`. For this, complete `one_hot_encoder(.)`

At the end of the task, your data will be ready to train the model designed. We can use these functions!

> The above operations are mandatory for our task. In addition to these, other preprocessing techniques can be performed.

## Design Segmentation Model

Before go on, please search and answer following questions:

- What is the difference between CNN and Fully CNN (FCNN) ?
- What are the different layers on CNN ?
- What is activation function ? Why is softmax usually used in the last layer?

There is a script to design our model: `model.py`. In this script, we could program our model. This will require research. Below are the videos we think can help you:

- [Create tensors in PyTorch + Matrix Multiplication](https://youtu.be/gXVQKueWoIA)
- [Forward Pass](https://youtu.be/bQ3vD_3WnLQ)
- [Backpropogation](https://youtu.be/HKWp78wEVJU)
- [Calculating Gradients](https://youtu.be/attSzjiD7YU)
- [Build NN & model.py](https://youtu.be/AojVTBLnwNM)

To visualize your model at the end, you can use [this](http://alexlenail.me/NN-SVG/) website.

## Train

Before go on, please search and answer following questions:

- What is parameter in NN ?
- What is hyper-parameter in NN ?
- We mention the dataset and we separate it into 2: training & test. In addition to them, there is a validation dataset. What is it for?
- What is epoch?
- What is batch?
- What is iteration? Explain with an example: "If we have x images as data and batch size is y. Then an epoch should run z iterations."
- What Is the Cost Function?
- The process of minimizing (or maximizing) any mathematical expression is called optimization. What is/are the purpose(s) of an optimizer in NN?
- What is Batch Gradient Descent & Stochastic Gradient Descent? Compare them.
- What is Backpropogation ? What is used for ?

We prepare a `train.py` script that combines all our work & techniques mention in the questions. Play with hyper-parameters and examine their effects! Enjoy 🙂