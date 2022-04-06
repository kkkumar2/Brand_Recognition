import os


def test_json_path(test_read_yaml):
    config = test_read_yaml
    prediction_dir = config['prediction']["prediction_dir"]
    jsonfile_dir = config["prediction"]["jsonfile_dir"] 
    jsonfile_name = config['prediction']["jsonfile_name"]
    jsonfile_path = os.path.join(prediction_dir,jsonfile_dir,jsonfile_name) 
    assert os.path.isfile(jsonfile_path) 


def test_pth_path(test_read_yaml):
    config = test_read_yaml
    prediction_dir = config['prediction']["prediction_dir"]
    save_model_dir = config["prediction"]["save_model_dir"]
    save_model_name = config["prediction"]["save_model_name"]
    save_mode_path = os.path.join(prediction_dir,save_model_dir,save_model_name)
    assert os.path.isfile(save_mode_path)

def test_config_path(test_read_yaml):
    config = test_read_yaml
    prediction_dir = config["prediction"]["prediction_dir"]
    config_dir = config["prediction"]["model_config_dir"] 
    output_config = config["prediction"]["output_config"]
    model_config_name = config["prediction"]["model_config"] 

    model_confgi_path = os.path.join(prediction_dir,config_dir,model_config_name)
    output_config_path = os.path.join(prediction_dir,config_dir,output_config)

    assert os.path.isfile(model_confgi_path)
    assert os.path.isfile(output_config_path)