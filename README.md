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
Let we dive into deeper about this json file. The important features of this json are <b>classTitle</b>(to determine which object properties we are seeing or basically the class name of the object), <b>points,exterior</b>(We're gonna create mask by this specific points). <b>Exterior</b> refers to list of coordinates of the points that make up the outer edge of the polygon. You can take a look example mask that i obtained in jupyter notebook using json2 mask script.![Json2Mask](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/664a03e3-254c-49f9-acb9-5cda517f6340)
![Json2MaskLine](https://github.com/BeytullahYayla/FordOtosan-L4Highway-Internship-Project/assets/78471151/e40e3008-0bd5-4b39-9404-4fcde68e24ba)

