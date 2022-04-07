import base64
import io
from PIL import Image
import cv2
import numpy as np
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import yaml
import os

#def error_handel_user_images(base64bytes:str):

#    base64str = base64.b64decode(base64bytes)   
#    bytesObj = io.BytesIO(base64str)
#    img = Image.open(bytesObj)
#    img_cv = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)



class ClientImageInput(BaseModel):
    image: bytes
    threshold: Optional[float] = 0.5
    IOR:Optional[str] = None
    float_center_crop:Optional[float] = 0.0
    image_crop_ratio: Optional[bool] = False
    x_axis:Optional[tuple] = (0.0,0.0)
    y_axis:Optional[tuple] = (0.0,0.0)


class ClientImageOutput(BaseModel):
    image: bytes


    
def read_yaml(filename:Path) -> dict:

    with open(filename , 'r') as config_file:
        data = yaml.safe_load(config_file)
    
    return data

class ModelLabelmapPath:
    @staticmethod
    def get_config_path(configfile_path:Path) -> dict:
        config = read_yaml(configfile_path)
        prediction_dir = config["prediction"]["prediction_dir"]
        labelmap_dir = config['prediction']["labelmap_dir"]
        ckpt_dir = config["prediction"]['ckpt_dir']
        labelmap_name = config['prediction']['labelmap_name']
#        model_name = config['prediction']['model_name']
        Ckpt_path = os.path.join(prediction_dir,labelmap_dir,ckpt_dir)
        labelmap_path = os.path.join(prediction_dir,labelmap_dir,labelmap_name)

#        return {"Path_To_Ckpt":Ckpt_path,"Labelmap_Path":labelmap_path}
        return Ckpt_path,labelmap_path