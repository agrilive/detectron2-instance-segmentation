# Retrain Mask R-CNN Model

#### Task

Modify the Mask R-CNN network to use MobileNet as a backbone.

#### Approach

I used the [Detectron2](https://github.com/facebookresearch/detectron2), a PyTorch implementation of Mask R-CNN model, in this task. These are the steps I took to retrain a MobileNet backbone model on my own dataset.

1. Create dataset
    - The steps used to create the dataset can be found in the [previous task](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset) that I have done.
2. Add MobileNet backbone
    - Clone the [VoVNetv2 Detectron2](https://github.com/youngwanLEE/vovnet-detectron2) repo, which added a VoVNetv2 backbone to the Detectron2 framework.
    - Add the config file of MobileNetv2 into the VoVNetv2 folder
3. Retrain model
    - Register the dataset in [Detectron2](https://github.com/facebookresearch/detectron2).
    - Set configurations and retrain model with pre-trained Faster R-CNN model.
    - Make predictions on test set.

## 1. Create dataset

Refer to the previous task done on [retraining a Mask R-CNN model on custom dataset](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset) for a detailed guide on how to create a custom dataset.

## 2. Add MobileNet backbone

From here onwards, I used Google Colab for the task.

### 2.1 Clone and install Detectron2 repo

Install torch and torchvision

```
!pip install -U torch torchvision
!pip install git+https://github.com/facebookresearch/fvcore.git
import torch, torchvision
torch.__version__
```

Install the Detectron2 repo

```
!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
!pip install -e detectron2_repo
```

### 2.2 Clone VoVNetv2 repo

The [VoVNetv2](https://github.com/youngwanLEE/vovnet-detectron2) repo compares its VoVNetv2 backbone with the MobileNetv2 results. Therefore, we can tap on the MobileNetv2 backbone that they have done.

```
!git clone https://github.com/youngwanLEE/vovnet-detectron2 vovnet_detectron2
```

### 2.3 Add MobileNetv2 config file

After cloning the repo, add the [MobileNetv2 config file](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/mobilenet-mask-rcnn/mask_rcnn_Mv2_FPNLite_3x.yaml) into the VoVNetv2 repo, `config` folder.

## 3. Retrain model

### 3.1 Register the dataset with Detectron2

Since the dataset I used is a custom one, I have to register it with Detectron2.

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

We are using the MobileNetv2 config file that we have added to the path `vovnet_detectron2/configs`. 

```
cfg = get_cfg()
cfg.merge_from_file("vovnet_detectron2/configs/mask_rcnn_Mv2_FPNLite_3x.yaml")
cfg.DATASETS.TRAIN = ("pandasH_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
```

Register the MobileNetv2 backbone.

```
from vovnet_detectron2.vovnet import mobilenet
```

Thereafter, train the model.

```
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

Unlike the Detectron2 implementation with ResNet backbone, the MobileNetv2 backbone pales in comparison on its ability to predict the instances of interest.

For instance, the giant pandas could not be detected. Upon closer inspection, you would see that the model is trying to predict instances of red pandas. 

![mobilenet_panda](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/mobilenet-mask-rcnn/sample-images/mobilenet_panda.png?raw=true))

For red pandas,, the model correctly predicted instances of red pandas. However, these masks were less accurate than those predicted with the ResNet backbone. In addition, the model predicted several instances of red pandas although there was only one instance of red panda.

![mobilenet_red_panda](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/mobilenet-mask-rcnn/sample-images/mobilenet_red_panda.png?raw=true))

Similarly, this was the case for two red pandas.

![mobilenet_red_panda_2](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/mobilenet-mask-rcnn/sample-images/mobilenet_red_panda_2.png?raw=true))


