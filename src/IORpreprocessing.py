import cv2 as cv
from nptyping import NDArray
from research.prediction import BrandsLog
from typing import Dict,ByteString,Tuple,Optional


class IOR(BrandsLog):
    def __init__(self,**kwargs):
        super(IOR, self).__init__(**kwargs)

    def xaxis(self,side:str,thresholds:float) -> Dict[str,ByteString] :

        img = self.base64toimage
        width = img.shape[1]
        middle_point = width//2 

        if side in "left":
            left_img = img[:,:middle_point]
            left_predict_img = self.getPredictions(left_img,thresholds)
            img[:,:middle_point] = left_predict_img # image concatenated happend
        
        elif side in 'right':
            right_img = img[:,middle_point:]
            right_predict_img = self.getPredictions(right_img,thresholds)
            img[:,middle_point:] = right_predict_img 


        base64img = self.imagetobase64(img) 

        return {"image": base64img}
    
    def yaxis(self,side:str ,thresholds:float) -> Dict[str,ByteString] :

        img = self.base64toimage
        height = img.shape[0]
        middle_point = height//2 
        
        if side in "top":
            top_img = img[:middle_point] 
            top_predict_img = self.getPredictions(top_img,thresholds)
            img[:middle_point] = top_predict_img
        
        elif side in "buttom":
            buttom_img = img[middle_point:]
            buttom_predict_img = self.getPredictions(buttom_img,thresholds)
            img[middle_point:] = buttom_predict_img
        
        base64img = self.imagetobase64(img) 

        return {"image": base64img}

    def crop(self,thresholds:float,percent_crop:float=0.0) ->Dict[str,ByteString]:

        img = self.base64toimage

        if percent_crop > 0.0:
            height,width,_ = img.shape
            cut_pixel_height_side = int(percent_crop*height)//2
            cut_pixel_width_side = int(percent_crop*width)//2

            crop_img = img[cut_pixel_height_side : -cut_pixel_height_side , cut_pixel_width_side : -cut_pixel_width_side]
            
            crop_predict_img = self.getPredictions(crop_img,thresholds)
            img[cut_pixel_height_side : -cut_pixel_height_side , cut_pixel_width_side : -cut_pixel_width_side] = crop_predict_img 
        
        else:
            normal_img = img
            img = self.getPredictions(normal_img,thresholds)
        
        base64img = self.imagetobase64(img)
        return {"image": base64img}

    def amount_cut_images(self,x_axis:Tuple[float,float]=None,y_axis:Tuple[float,float]=None,thresholds:Optional[float]=0.3) -> Dict[str,ByteString]:
        """

        Args:
            x_axis (Tuple[float,float]) = (xmin,xmax)
            y_axis (Tuple[float,float]) = (ymin,ymax)
            thresholds:Optional[float]=0.3

        """
        img = self.base64toimage
        
        if isinstance(x_axis,Tuple) or isinstance(y_axis,Tuple):
            height,width,_ = img.shape

            if isinstance(x_axis,tuple):
                start_pixel_width = x_axis[0]
                end_pixel_width = x_axis[1]
                cut_img_width = img[ : ,   start_pixel_width: end_pixel_width]

            if isinstance(y_axis,tuple):
                start_pixel_height = y_axis[0]
                end_pixel_height = y_axis[1]

                if isinstance(x_axis,tuple):
                    cut_img_height_width =   cut_img_width[start_pixel_height: end_pixel_height]
                    cut_img = self.getPredictions(cut_img_height_width,thresholds)
                else:
                    cut_img_height = img[  start_pixel_height:   end_pixel_height]
                    # cv.imwrite('width.jpg', cut_img_height)
                    cut_img = self.getPredictions(cut_img_height,thresholds)

            elif isinstance(x_axis,tuple):
                cut_img = self.getPredictions(cut_img_width,thresholds)
                
            # cv.imwrite("cut.jpg",cut_img)

            if isinstance(y_axis,tuple):
                if isinstance(x_axis,tuple):
                    cut_img_width[start_pixel_height: end_pixel_height] = cut_img
                    img[ : ,   start_pixel_width: end_pixel_width] = cut_img_width
                else:
                    img[  start_pixel_height:   end_pixel_height] = cut_img
            elif isinstance(x_axis,tuple):
                img[ : ,   start_pixel_width: end_pixel_width] = cut_img

        else:
            normal_img = img
            img = self.getPredictions(normal_img)
        
        
        base64img = self.imagetobase64(img)

        return {"image": base64img}
