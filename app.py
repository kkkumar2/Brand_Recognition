
from fastapi import FastAPI ,Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.responses import HTMLResponse,FileResponse,JSONResponse
from typing import Optional,List,Union,Dict
from pathlib import Path
from utils.all_utills import ClientImageInput,ClientImageOutput, ModelLabelmapPath
from research.prediction import BrandsLog
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

Pathmodellabelmap = ModelLabelmapPath.get_config_path(os.path.join("config",'config.yaml'))
class ClientApp(BrandsLog):
    def __init__(self,**kwargs):
        super(ClientApp, self).__init__(**kwargs)


clApp = ClientApp(**Pathmodellabelmap)


# templates = Jinja2Templates(directory="webapp/templates")

# @app.get("/",response_class=HTMLResponse)
# def read(request:Request):
#     return templates.TemplateResponse("index.html",{"request":request})


@app.post("/predict",response_model=List[Union[ClientImageOutput,ClientImageInput]])
def predict(file:ClientImageInput):
    if not isinstance(file.image ,bytes):
        raise NotEncodeBase64(message="image not in enocde bytes format" )
    elif isinstance(file.image,bytes):
        try:
            clApp.base64toimage = file.image
        except :
            raise ImageIsNotOpening(message="image is Not opening")

    
    output = clApp.getPredictions()
    return output
    
if __name__ == "__main__":
    uvicorn.run(app,port=8080)