
from src.prediction import BrandsLog
import pytest
import os
import numpy as np


logo =None

def test_brandLog(test_read_yaml):
    config = test_read_yaml
    prediction_dir = config["prediction"]["prediction_dir"]
    jsonfile_dir = config["prediction"]["jsonfile_dir"] 
    jsonfile_name = config['prediction']["jsonfile_name"] 
    config_dir = config["prediction"]["model_config_dir"] 
    output_config = config["prediction"]["output_config"]
    model_config_name = config["prediction"]["model_config"] 
    save_model_dir = config["prediction"]["save_model_dir"]
    save_model_name = config["prediction"]["save_model_name"]

    jsonfile_path = os.path.join(prediction_dir,jsonfile_dir,jsonfile_name) 
    model_confgi_path = os.path.join(prediction_dir,config_dir,model_config_name)
    output_config_path = os.path.join(prediction_dir,config_dir,output_config)
    save_mode_path = os.path.join(prediction_dir,save_model_dir,save_model_name)

    global logo
    logo = BrandsLog(save_mode_path,model_confgi_path,output_config_path,jsonfile_path)

def test_base64toimage(test_create_random_base64image):
    logo.base64toimage = test_create_random_base64image
    array = logo.base64toimage
    assert isinstance(array,np.ndarray)
    assert array.dtype == np.dtype(np.uint8)


def test_inference():

    output = logo.inference(logo.base64toimage,threshold=0.3)
    assert isinstance(output,np.ndarray)
    assert output.dtype == np.dtype(np.uint8)