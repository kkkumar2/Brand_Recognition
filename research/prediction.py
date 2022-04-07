from research.object_detection.utils import label_map_util
from research.object_detection.utils import visualization_utils as vis_util
import os
import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as  np
import base64
import glob
import cv2
import io
import re 

from typing import Union,Optional,Any,List,ByteString
from nptyping import NDArray


class BrandsLog:
    def __init__(self,Path_To_Ckpt:Path,Labelmap_Path:Path) -> None:


        labelmap = label_map_util.load_labelmap(Labelmap_Path)
        self.categories = label_map_util.convert_label_map_to_categories(labelmap,
                                                                         use_display_name=True
                                                                    # ,max_num_classes = self._Num_Classes_Label_map(Labelmap_Path) 
                                                                    )
        #  I have successful change the source code i have don't need to given max_num_classes 

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(Path_To_Ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    @property
    def base64toimage(self) -> NDArray[np.uint8]  :

        img_cv = cv2.cvtColor(np.array(self.imagePil),cv2.COLOR_RGB2BGR) #https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format      
        return img_cv

    @base64toimage.setter
    def base64toimage(self,img_strings:Optional[bytes]) :
        base64str = base64.b64decode(img_strings)
        bytesObj = io.BytesIO(base64str) # we used self bytesObj because it size of bytes is less than cv2 so we used 
        self.imagePil = Image.open(bytesObj)

    def imagetobase64(self,ImageArray:NDArray[Any]) -> ByteString:
        
        img_RGB = cv2.cvtColor(ImageArray,cv2.COLOR_BGR2RGB)
        image_array = Image.fromarray(img_RGB)
        buffered = io.BytesIO()
        image_array.save(buffered,format="JPEG")
        bas64str = base64.b64encode(buffered.getvalue()).decode('utf-8') #https://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string
        buffered.flush()
        return bas64str

    # def _Num_Classes_Label_map(self,LabelMapPath:Path) -> int:
    #     with open(LabelMapPath,'r') as f:
    #         data = f.read()
    #     total_classes = re.findall(r"\d+",data)[-1]
    #     return int(total_classes)

    def getPredictions(self,image_array:NDArray,threshold:float= 0.7):
        
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        category_index = label_map_util.create_category_index(self.categories)
        sess = tf.Session(graph=self.detection_graph)

        image =  image_array
        image_expand = np.expand_dims(image,axis=0)


        (boxes,scores,classes,num) = sess.run([detection_boxes,detection_scores
                                            ,detection_classes,num_detections]
                                           ,feed_dict={image_tensor:image_expand})
            
        vis_util.visualize_boxes_and_labels_on_image_array(
            image
            ,np.squeeze(boxes)
            ,np.squeeze(classes).astype(np.int32)
            ,np.squeeze(scores)
            ,category_index
            ,use_normalized_coordinates=True
            ,line_thickness=10 
            ,min_score_thresh = threshold

            )
        
        cv2.imwrite("output4.jpg",image)
    

        return image


