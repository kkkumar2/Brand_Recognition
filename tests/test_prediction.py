
from research.prediction import BrandsLog
import pytest
import os
import numpy as np





logo =None

def test_brandLog(test_read_yaml):
    config = test_read_yaml
    prediction_dir = config['prediction']["prediction_dir"]
    labelmap_dir = config['prediction']['labelmap_dir']
    labelmap_name = config['prediction']['labelmap_name']
    ckpt_dir = config['prediction']["ckpt_dir"]
    model_name = config['prediction']["model_name"]
    labelmap_path = os.path.join(prediction_dir, labelmap_dir,labelmap_name)
    ckpt_path = os.path.join(prediction_dir,ckpt_dir,model_name)

    global logo
    logo = BrandsLog(ckpt_path,labelmap_path)

def test_base64toimage(test_create_random_base64image):
    logo.base64toimage = test_create_random_base64image
    array = logo.base64toimage
    assert isinstance(array,np.ndarray)
    assert array.dtype == np.dtype(np.uint8)


#def test_getPredictions():
#
#    output = logo.getPredictions()
#    assert isinstance(output,list)
#   assert "image" in output[-1]    