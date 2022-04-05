
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
import logging
import time
import os

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




class ClientApp(BrandsLog):
    def __init__(self,Path_Ckpt:Path,labelmap_ph:Path):
        super(ClientApp, self).__init__(Path_Ckpt,labelmap_ph)

#path2 = os.path.join('prediction_service','my_model','labelmap.pbtxt')
#path1 = os.path.join('prediction_service','my_model','saved_model')
#clApp = ClientApp(path1,path2)
Pathmodellabelmap = ModelLabelmapPath.get_config_path(os.path.join("config",'config.yaml'))
clApp = ClientApp(**Pathmodellabelmap)

print("Loggon started")
logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir = 'logs_dir'
standard_logs = 'general_logs'
os.makedirs(logs_dir, exist_ok=True)
general_logs_dir = os.path.join(logs_dir, standard_logs)
os.makedirs(general_logs_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(general_logs_dir, 'app.log'), level=logging.INFO, format=logging_str,filemode='a')
logging.info("API hitted")

templates = Jinja2Templates(directory="webapp/templates")

#@app.get("/",response_class=HTMLResponse)
#def read(request:Request):
#    return templates.TemplateResponse("index.html",{"request":request})



#@app.post("/predict",response_model=List[Union[ClientImageOutput,ClientImageInput]])
#def predict(file:UploadFile=File(...)):
#@app.post("/predict")

@app.post("/predict",response_model=ClientImageInput)
def predict(file:ClientImageInput):
    logging.info("Predict API hitted")
#    s = file.image
    if not isinstance(file.image ,bytes):
        raise NotEncodeBase64(message="image not in enocde bytes format" )
    elif isinstance(file.image,bytes):
        try:
#            error_handel_user_images(file.image)
            clApp.base64toimage = file.image
        except :
            raise ImageIsNotOpening(message="image is Not opening")

#    clApp.base64toimage = file.image
    logging.info("calling  getPredictions")
    start = time.time()
    output = clApp.run_inference()
    end = time.time()
    logging.info(f"total time is:{end-start}")
    logging.info(f"output recieved back:{output}")
    return output
    
    # return {'sandee':6}
    
if __name__ == "__main__":
    uvicorn.run(app,port=8080)