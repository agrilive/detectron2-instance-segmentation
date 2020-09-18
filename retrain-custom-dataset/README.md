# Retrain Mask R-CNN Model

#### Task

Retrain the Mask R-CNN model to detect an object of your choice

#### Approach

I used the [Detectron2](https://github.com/facebookresearch/detectron2), a PyTorch implementation of Mask R-CNN model, in this task. These are the steps I took to retrain the model on my own dataset.

1. Create dataset
    - I used my own images, made up of giant panda and red panda images. 
    - [labelme](https://github.com/wkentaro/labelme) was used to annotate the images.
    - [labelme2coco](https://github.com/Tony607/labelme2coco) was used to convert the labelme data format to COCO data format.
2. Retrain model
    - Register the dataset in [Detectron2](https://github.com/facebookresearch/detectron2).
    - Set configurations and retrain model with pre-trained Faster R-CNN model.
    - Make predictions on test set.

## 1. Create dataset

#### Training dataset

My training dataset contains 57 images, made up of giant panda and red panda images. These are images that I took on a trip to China.

The Mask R-CNN model was trained on the COCO dataset, which is made up of 80 object categories. There are several animal categories, including bear. I explored the [COCO dataset](https://cocodataset.org/#explore) and found out that the bear images include several species of bears, including polar bears, brown bears and grizzly bears.

Since I will be using the COCO weights to retrain the model, I did not have to use a large dataset. 

#### Test dataset

My test dataset contains 20 images, made up of giant panda and red panda images as well. These are images taken from Google Images, with creative license enabled.

### 1.1 Label images

[labelme](https://github.com/wkentaro/labelme) was used to annotate the dataset. I followed the installation instructions for Ubuntu.

```
sudo apt-get install python-pyqt5 
sudo pip install labelme
```

The output file is a `.json` file of the same name. Keep the `.json` file in the same folder as your images.

Sample image:

![labelme_panda](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset/sample-images/labelme_panda.png?raw=true))

### 1.2 Convert annotations to COCO data format

I used Google Colab for the codes from here.

Clone the labelme2coco repo and install `labelme`.

```
!git clone https://github.com/Tony607/labelme2coco.git
!pip install labelme
```

Then, use these commands to convert the labelme data format to COCO data format. 

```
!python labelme2coco/labelme2coco.py data/train --output=data/train.json
!python labelme2coco/labelme2coco.py data/test --output=data/test.json
```

## 2. Retrain model

### 2.1 Register the dataset with Detectron2

To use a dataset with Detectron2, you need to register the dataset.

Here, since my images were of different sizes, I created a custom `get_image_dim` function that uses `OpenCV` to get the height and width of the images.

```
def get_image_dim(directory, json_file):
  imagename = json_file.replace('json', 'jpg')
  img = cv2.imread(imagename,)
  height, width = img.shape[:2]
  return height, width

def get_dataset_dicts(directory):
    classes = ['panda', 'red_panda']
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        height, width = get_image_dim(directory, json_file)
        record["height"] = height#600
        record["width"] = width#800
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "test"]:
    DatasetCatalog.register("pandasH_" + d, lambda d=d: get_dataset_dicts('data/' + d))
    MetadataCatalog.get("pandasH_" + d).set(thing_classes=['panda', 'red_panda'])
```

Thereafter, retrieve the metadata.

```
dataset_metadata = MetadataCatalog.get("pandasH_train")
```

### 2.2 Set configurations

I configured the parameters as follows:

```
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("pandasH_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
```

### 2.3 Make predictions on test set

```
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("pandasH_test", )
predictor = DefaultPredictor(cfg)

dataset_dicts = get_dataset_dicts('data/test')
for d in random.sample(dataset_dicts, 10):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=dataset_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (10, 7))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()
```

## Results

Generally, the model is able to predict the pandas rather accurately.

![results_panda](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset/sample-images/results_panda.png?raw=true))

The model is able to differentiate between the tree branch and the red panda! Amazing indeed.

![results_red_panda](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset/sample-images/results_red_panda.png?raw=true))

However, several red panda images could not be predicted accurately. The model seems to think that the red panda's tail is another panda instance.

![results_red_panda_poor](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset/sample-images/results_red_panda_poor.png?raw=true))


