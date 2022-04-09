
from fastapi import FastAPI ,Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.responses import HTMLResponse,FileResponse,JSONResponse
from typing import Optional,List,Union,Dict
from pathlib import Path
from src.utils.all_utills import ClientImageInput,ClientImageOutput, ModelLabelmapPath
from src.IORpreprocessing import IOR
import os,logging,time


logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(logs_dir, 'app.log'), level=logging.INFO, format=logging_str,filemode='a')


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

Pathmodellabelmap = ModelLabelmapPath.get_config_path(os.path.join("config",'config.yaml'))
class ClientApp(IOR):
    def __init__(self,**kwargs):
        super(ClientApp, self).__init__(**kwargs)


cliApp = ClientApp(**Pathmodellabelmap)
logging.info("clientapp start")

# templates = Jinja2Templates(directory="webapp/templates")

# @app.get("/",response_class=HTMLResponse)
# def read(request:Request):
#     return templates.TemplateResponse("index.html",{"request":request})


@app.post("/predict",response_model=ClientImageOutput)
def predict(file:ClientImageInput):
    since = time.time()
    logging.info("Predict API hitted")


    if not isinstance(file.image ,bytes):
        logging.info("image is not in bytes format")
        raise NotEncodeBase64(message="image not in enocde bytes format" )

    elif isinstance(file.image,bytes):
        try:
            cliApp.base64toimage = file.image
        except :
            logging.info("Image is not opening")
            raise ImageIsNotOpening(message="image is Not opening")

    if file.IOR in ['left','right']:
        output = cliApp.xaxis(file.IOR,file.threshold)
    
    elif file.IOR in ['top','buttom']:
        output = cliApp.yaxis(file.IOR,file.threshold)
    
    elif file.image_crop_manual :
        output = cliApp.amount_cut_images(file.x_axis,file.y_axis,file.threshold) #,x_axis=(220,357),y_axis=(120,185)x_axis=(230,340),y_axis=(100,200)
    else:
        output = cliApp.crop(file.threshold,file.float_center_crop)

    time_elapsed = time.time() - since
    logging.info(f"Time is Taken  ::: {time_elapsed % 60:.0f}s imageIOR in mode {file.IOR} ")
    logging.info(f"{'-'*10}> Output give back to user <{'-'*10}")
    return output

    
if __name__ == "__main__":
    uvicorn.run(app,port=8080)