import os
import cv2
from PIL import Image
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

class Detector:

	def __init__(self):

		self.cfg = get_cfg()
		self.cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		self.cfg.DATASETS.TRAIN = ("pandasH_train",)
		self.cfg.DATASETS.TEST = ()
		self.cfg.DATALOADER.NUM_WORKERS = 2
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		self.cfg.SOLVER.IMS_PER_BATCH = 2
		self.cfg.SOLVER.BASE_LR = 0.00025
		self.cfg.SOLVER.MAX_ITER = 1000
		self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
		self.cfg.MODEL.DEVICE = "cpu"
		self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
		self.cfg.DATASETS.TEST = ("pandasH_test", )

	def inference(self, file):
		predictor = DefaultPredictor(self.cfg)
		im = cv2.imread(file)
		outputs = predictor(im)
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
		v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))
		return img