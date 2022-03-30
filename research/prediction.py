from research.object_detection.utils import label_map_util
from research.object_detection.utils import visualization_utils as vis_util
from research.object_detection.utils import ops as utils_ops
import os
import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as  np
import base64
import glob
import cv2
from typing import Union,Optional,Any,List
import io
from nptyping import NDArray
class BrandsLog:
    def __init__(self,Path_To_Ckpt:Path,labelmap_path:Path) -> None:


        labelmap = label_map_util.load_labelmap(labelmap_path)
        self.categories = label_map_util.convert_label_map_to_categories(labelmap
                                                                    ,max_num_classes = 6
                                                                    ,use_display_name=True)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(Path_To_Ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    @property
    def base64toimage(self) -> NDArray[np.unit8]  :

        img = Image.open(self.bytesObj)
        del self.bytesObj
        img_cv = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR) #https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
        
        return img_cv

    @base64toimage.setter
    def base64toimage(self,img_strings:Optional[bytes]) :
        base64str = base64.b64decode(img_strings)
        self.bytesObj = io.BytesIO(base64str) # we used self bytesObj because it size of bytes is less than cv2 so we used 


        

    def getPredictions(self):
        
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        category_index = label_map_util.create_category_index(self.categories)
        class_name_mapping = {item["id"]:item["name"] for item in self.categories}
        min_score_thresh = 0.30 

        sess = tf.Session(graph=self.detection_graph)

        image =  self.base64toimage
        image_expand = np.expand_dims(image,axis=0)
        del self.base64toimage

        (boxes,scores,classes,num) = sess.run([detection_boxes,detection_scores
                                            ,detection_classes,num_detections]
                                           ,feed_dict={image_tensor:image_expand})

        result = scores.flatten()
        res = [idx for idx,clss_perc in enumerate(result) if clss_perc >0.40]

        top_class = classes.flatten()

        res_list = [top_class[i] for i in res]
        clss_name = [class_name_mapping[i] for i in res_list]

        top_score = [i for i in result if i > min_score_thresh]
        new_box = boxes.squeeze()
        max_boxes_to_draw = new_box.shape[0]

        listofoutput = []

        for name,score,i in zip(clss_name,top_score,range(min(max_boxes_to_draw,new_box.shape[0]))):
            valDict = {}
            valDict['className'] = name 
            valDict['confidence'] = str(score) 

            if result is None or result[i] > min_score_thresh:
                val = list(new_box[i] )
                valDict["yMin"] = str(val[0])
                valDict['xMin'] = str(val[1])
                valDict['yMax'] = str(val[2])
                valDict['xMax'] = str(val[3])
                listofoutput.append(valDict)
            
        vis_util.visualize_boxes_and_labels_on_image_array(
            image
            ,np.squeeze(boxes)
            ,np.squeeze(classes).astype(np.int32)
            ,np.squeeze(scores)
            ,category_index
            ,use_normalized_coordinates=True
            ,line_thickness=10 
            ,min_score_thresh = 0.60

            )
        
        cv2.imwrite("output4.jpg",image)



