
from fastapi import FastAPI ,Request, HTTPException, UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.responses import HTMLResponse,FileResponse,JSONResponse
from typing import Optional,List,Union,Dict
from pathlib import Path
from utils.all_utils import ClientImageInput,ClientImageOutput,ModelLabelmapPath
from src.prediction import BrandsLog
from src.IORpreprocessing import IOR
import logging
import time
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir = 'logs_dir'
standard_logs = 'general_logs'
os.makedirs(logs_dir, exist_ok=True)
general_logs_dir = os.path.join(logs_dir, standard_logs)
os.makedirs(general_logs_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(general_logs_dir, 'app.log'), level=logging.INFO, format=logging_str,filemode='a')


app  = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST","GET"], 
    allow_headers=["*"],
    max_age=2 # how mcuh hit api per second
    )


class NotEncodeBase64(Exception):
    def __init__(self,message:str=None):
        self.message = message

class ImageIsNotOpening(Exception):
    def __init__(self,message:str=None):
        self.message = message

@app.exception_handler(NotEncodeBase64)
def not_encode_base64(request:Request,exc:NotEncodeBase64):
    
    return JSONResponse(
        status_code=418
        ,content = {"message":f"{exc.message} "}
    )

@app.exception_handler(ImageIsNotOpening)
def image_not_open(request:Request , exc:ImageIsNotOpening):
    
    return JSONResponse(
        status_code=418
        ,content = {"message":f"{exc.message}"}
    )

Path_To_Ckpt,Labelmap_Path = ModelLabelmapPath.get_config_path(os.path.join("config",'config.yaml'))
class ClientApp(IOR):
    def __init__(self,Path_Ckpt:Path,labelmap_ph:Path):
        super(ClientApp, self).__init__(Path_Ckpt,labelmap_ph)

clApp = ClientApp(Path_To_Ckpt,Labelmap_Path)

@app.post("/predict",response_model=ClientImageOutput)
def predict(file:ClientImageInput):
    
    since = time.time()
    logging.info("Predict API hitted")

    if not isinstance(file.image ,bytes):
        raise NotEncodeBase64(message="image not in enocde bytes format" )

    elif isinstance(file.image,bytes):
        try:
            clApp.base64toimage = file.image
        except :
            logging.info("image is Not opening")
            raise ImageIsNotOpening(message="image is Not opening")

    if file.IOR in ['left','right']:
        output = clApp.xaxis(file.IOR,file.threshold)
    
    elif file.IOR in ['top','buttom']:
        output = clApp.yaxis(file.IOR,file.threshold)
    elif file.image_crop_ratio:
        output = clApp.amount_cut_images(file.x_axis,file.y_axis)
    else:
        output = clApp.crop(file.threshold,file.float_center_crop)

    time_elapsed = time.time() - since
    logging.info(f"Time is Taken  ::: {time_elapsed % 60:.0f}s imageIOR in mode {file.IOR} ")
    logging.info(f"{'-'*10}> Output give back to user <{'-'*10}")

    return output

    
if __name__ == "__main__":
    uvicorn.run(app,port=8080)