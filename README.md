# FordOtosan-L4Highway-Internship-Project
## 🚩 Contents
- [Project Structure](#-project-structure)
- [Topics](#-pages)
  * [General Overview to Project Topics](https://sway.office.com/GWVpEgbbvFCstcKv?ref=Link)
- [Getting Started](#Getting-Started)
  *  [Json2Mask](#json2mask)
  *  [Mask On Image](#-mask-on-image)
  *  [Preprocessing](#-preprocessing)

 
## 🗃 Project Structure
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

## Getting Started
### The purpose of the project
In this project my aim is to detect drivable areas in highways using state of the art deep learning techniques and present a repository to autonomous vehicle engineers.

### Json2Mask
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


### Mask on Image
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

