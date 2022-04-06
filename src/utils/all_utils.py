from pathlib import Path
from pydantic import BaseModel
import yaml
import os
from typing import Optional


class ClientImageInput(BaseModel):
    image: bytes
    threshold: Optional[float] = 0.5
    IOR:Optional[str] = None
    float_center_crop:Optional[float] = 0.0


class ClientImageOutput(BaseModel):
    image: bytes



def read_yaml(filename:Path) -> dict:

    with open(filename , 'r') as config_file:
        data = yaml.safe_load(config_file)
    
    return data

# 'prediction_service\\model_config\\config.yaml'
class AllPath:
    @staticmethod
    def get_all_file_path(configfile_path:Path) -> dict:
        config = read_yaml(configfile_path)
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

        assert isinstance(jsonfile_path,(str,os.PathLike)), "train/test .json file"
        assert isinstance(model_confgi_path,(str,os.PathLike)), "model config.yaml"
        assert isinstance(output_config_path,(str,os.PathLike)), " train config.yaml"
        assert isinstance(save_mode_path,(str,os.PathLike)), "save model .pth"


        path_dict = {"Path_Of_pth" : save_mode_path
                    ,"Model_Config" : model_confgi_path
                    ,"Train_Config" : output_config_path
                    ,"Json_file" : jsonfile_path
                    }
        
        return path_dict

if __name__ == "__main__":
    s = AllPath().get_all_file_path("config/config_path.yaml")
    print(s)