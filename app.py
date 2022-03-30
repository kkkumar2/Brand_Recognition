
from fastapi import FastAPI ,Request,UploadFile,File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.responses import HTMLResponse,FileResponse
from typing import Optional,List,Union,Dict
from pathlib import Path
from pydantic import BaseModel
from research.prediction import BrandsLog




app  = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"], 
    allow_headers=["*"],
    max_age=2 # how mcuh hit api per second
    )


class ClientImageInput(BaseModel):
    image: bytes
    # threshold: Optional[float] = 0.4


class ClientImageOutput(BaseModel):
    className:str  
    confidence:str  
    yMin:str
    xMin:str
    yMax:str
    xMax:str



class ClientApp(BrandsLog):
    def __init__(self,Path_Ckpt:Path,labelmap_ph:Path):
        super(ClientApp, self).__init__(Path_Ckpt,labelmap_ph)


# templates = Jinja2Templates(directory="webapp/templates")

# @app.get("/",response_class=HTMLResponse)
# def read(request:Request):
#     return templates.TemplateResponse("index.html",{"request":request})


@app.post("/predict",response_model=List[Union[ClientImageOutput,ClientImageInput]])
async def predict(file:ClientImageInput):
    clApp.base64toimage = file.image
    output = clApp.getPredictions()
    return output
    
    # return {'sandee':6}
    
if __name__ == "__main__":
    clApp = ClientApp("prediction_service\\save_model\\frozen_inference_graph.pb","prediction_service\\labelmap\\labelmap.pbtxt")
    uvicorn.run(app,port=8080)