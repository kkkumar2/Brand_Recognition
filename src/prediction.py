import os
import tensorflow as tf
from pathlib import Path
from PIL import Image
import base64
import io
import re 
import logging
from typing import Union,Optional,Any,List,ByteString
from nptyping import NDArray
import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import pathlib
import glob
import matplotlib.pyplot as plt
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile




class BrandsLog:
    def __init__(self,Path_To_Ckpt:Path,labelmap_path:Path) -> None:


        labelmap = label_map_util.load_labelmap(labelmap_path)
#        self.categories = label_map_util.convert_label_map_to_categories(labelmap
#                                                                    ,max_num_classes = self._Num_Classes_Label_map(labelmap_path) 
#                                                                    ,use_display_name=True)
#        the above thing is returning a list with a dict i need a dict with a dict

        self.categories = label_map_util.create_category_index_from_labelmap(labelmap_path,
                                                                                 use_display_name=True)
        # we can direct extract num class label map "labelmap" variable but it take time for me debugging code so i have use custom function but it not optimally way
        
        self.model = tf.saved_model.load(Path_To_Ckpt)

    @property
    def base64toimage(self) -> NDArray[np.uint8]  :

        image = Image.open(self.bytesObj)
        del self.bytesObj
#        img_cv = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR) #https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        
#        return img_cv

    @base64toimage.setter
    def base64toimage(self,img_strings:Optional[bytes]) :
        base64str = base64.b64decode(img_strings)
        self.bytesObj = io.BytesIO(base64str) # we used self bytesObj because it size of bytes is less than cv2 so we used 


    def _imagetobase64(self,ImageArray:NDArray[Any]) -> ByteString:
        
        img_RGB = cv2.cvtColor(ImageArray,cv2.COLOR_BGR2RGB)
        image_array = Image.fromarray(img_RGB)
        buffered = io.BytesIO()
        image_array.save(buffered,format="png")
        bas64str = base64.b64encode(buffered.getvalue()).decode('utf-8') #https://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string
        
        return bas64str

    def _Num_Classes_Label_map(self,LabelMapPath:Path) -> int:
        with open(LabelMapPath,'r') as f:
            data = f.read()
        total_classes = re.findall(r"\d+",data)[-1]
        return int(total_classes)


        
        
        
    def run_inference_for_single_image(self, model, image):
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        output_dict = model(input_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict

    def run_inference(self):
        logging.info("Predictions started")
        image =  self.base64toimage
#        image_np = np.expand_dims(image,axis=0)
        # Actual detection.
        model = self.model
        output_dict = self.run_inference_for_single_image(model, image)
        category_index = self.categories
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        output_filename = 'output.jpg'
        cv2.imwrite(output_filename, image)
#        ListOfOutput = []
        ListOfOutput={"image":self._imagetobase64(image)}
        logging.info("Returning from predictions")

        return ListOfOutput



