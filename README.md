# Instance Segmentation with Detectron2

(September 2020)

I tried out three tasks related to instance segmentation with Mask R-CNN.

1. Retrain the Detectron2 model with a custom dataset.
2. Deploy model (1) with REST API using the Flask framework.
3. Change the backbone of the Mask R-CNN model to MobileNetv2 and retrain on the custom dataset. Compare results from (1) and (3).

## Task 1: Retrain the Detectron2 model with a custom dataset

The task and approach is detailed in the [retrain-custom-dataset](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset) folder. 

I labelled a pandas dataset, where the training images were made up of images I had taken when I was in China, and the test images were made up of images from Google. There were two classes: Giant Pandas and Red Pandas.

### Results

With a small dataset size of less than 60 images, the results were suprisingly accurate. This might be attributed to the bear images that were in the COCO dataset, whose model weights were used for retraining.

![results_panda](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset/sample-images/results_panda.png?raw=true))

![results_red_panda_poor](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/retrain-custom-dataset/sample-images/results_red_panda_poor.png?raw=true))

## Task 2: Deploy model (1) with REST API using the Flask framework

More information can be found in the [flask-api](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/flask-api) folder. 

An image of choice is posted and the model makes its inference. An image with the drawn mask is saved.

### Results

Image posted:

![t08](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/flask-api/data/t08.jpg?raw=true))

Resulting image:

![result](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/flask-api/data/result.png?raw=true))

## Task 3: Change the backbone of the Mask R-CNN model to MobileNetv2 and retrain on the custom dataset

Details can be found in the [mobilenet-mask-rcnn]() folder.

The common backbone used in the Mask R-CNN model is the ResNet architecture. I replaced it with the MobileNetv2 architecture, and retrain the Mask R-CNN model on the pandas dataset. 

### Results

As compared to the model with ResNet backbone, the model with MobileNetv2 backbone pales in comparison. It is a reflection of how the ResNet model gives a higher accuracy score than the MobileNetv2 model on the ImageNet dataset.

![mobilenet_panda](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/mobilenet-mask-rcnn/sample-images/mobilenet_panda.png?raw=true))

![mobilenet_red_panda_2](https://github.com/agrilive/detectron2-instance-segmentation/blob/master/mobilenet-mask-rcnn/sample-images/mobilenet_red_panda_2.png?raw=true))