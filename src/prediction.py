import cv2 as cv
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pathlib import Path
import io
import base64
from typing import Optional,ByteString,Any
from nptyping import NDArray
from collections import namedtuple


class BrandsLog:

	def __init__(self,Path_Of_pth:Path,Model_Config:Path,Train_Config:Path,Json_file:Path):

		# set model and test set
		self.model = Model_Config

		# obtain detectron2's default config
		self.cfg = get_cfg() 

		# load values from a file


		self.cfg.merge_from_file(Train_Config)

		#self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model))
		if torch.cuda.is_available():
			device = "cuda"
		else:
			device = "cpu"
		# set device to cpu
		self.cfg.MODEL.DEVICE = device

		# get weights 
		# self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model) 
		#self.cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
		self.cfg.MODEL.WEIGHTS = Path_Of_pth


		# build model from weights
		# self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

		coco_api = COCO(Json_file)
		cat_ids = sorted(coco_api.getCatIds())
		cats = coco_api.loadCats(cat_ids)
		self.class_name_list = [name['name'] for name in cats]


	@property
	def base64toimage(self) -> NDArray[np.uint8]  :

		img_cv = cv.cvtColor(np.array(self.imagePil),cv.COLOR_RGB2BGR) #https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format      
		# cv.imwrite("input.jpg",img_cv)
		return img_cv

	@base64toimage.setter
	def base64toimage(self,img_strings:Optional[bytes]) :
		base64str = base64.b64decode(img_strings)
		bytesObj = io.BytesIO(base64str) # we used self bytesObj because it size of bytes is less than cv2 so we used 
		self.imagePil = Image.open(bytesObj)

	def imagetobase64(self,ImageArray:NDArray) -> ByteString:
		
		# img_RGB = cv.cvtColor(ImageArray,cv.COLOR_BGR2RGB)
		img_RGB = ImageArray
		# cv.imwrite('color_img.jpg', img_RGB)
		image_array = Image.fromarray(img_RGB)
		buffered = io.BytesIO()
		image_array.save(buffered,format="JPEG")
		bas64str = base64.b64encode(buffered.getvalue()).decode('utf-8') #https://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string
		buffered.flush()
		return bas64str



	def inference(self,images_array:NDArray,threshold:float=0.5) -> NDArray:
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

		predictor = DefaultPredictor(self.cfg)
		im = images_array

		outputs = predictor(im)
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

		# visualise
		v = Visualizer(im, metadata=metadata, scale=1)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"),self.class_name_list)
		predicted_image = v.get_image()

		cv.imwrite("output.jpg",predicted_image)

		return predicted_image




