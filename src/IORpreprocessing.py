import cv2 as cv
from nptyping import NDArray
from src.prediction import BrandsLog
from typing import Dict,ByteString,Tuple


class IOR(BrandsLog):
    def __init__(self,Path_Ckpt,labelmap_ph):
        super(IOR, self).__init__(Path_Ckpt,labelmap_ph)

    def xaxis(self,ROI:str,thresholds:float) -> Dict[str,ByteString] :

        img = self.base64toimage
        width = img.shape[1]
        middle_point = width//2
        blurred_img = cv.GaussianBlur(img, (21, 21), 0) 

        if ROI in "left":
            left_img = img[:,:middle_point]
            left_predict_img = self.run_inference(left_img,thresholds)
#            img[:,:middle_point] = left_predict_img # image concatenated happend blurring the unselected ROI 
            blurred_img[:,:middle_point] = left_predict_img 
        
        elif ROI in 'right':
            right_img = img[:,middle_point:]
            right_predict_img = self.run_inference(right_img,thresholds)
#            img[:,middle_point:] = right_predict_img 
            blurred_img[:,middle_point:] = right_predict_img 


        base64img = self._imagetobase64(blurred_img) 

        return {"image": base64img}
    
    def yaxis(self,ROI:str ,thresholds:float) -> Dict[str,ByteString] :

        img = self.base64toimage
        height = img.shape[0]
        middle_point = height//2 
        blurred_img = cv.GaussianBlur(img, (21, 21), 0) 
        
        if ROI in "top":
            top_img = img[:middle_point] 
            top_predict_img = self.run_inference(top_img,thresholds)
#            img[:middle_point] = top_predict_img
            blurred_img[:middle_point] = top_predict_img
        
        elif ROI in "buttom":
            buttom_img = img[middle_point:]
            buttom_predict_img = self.run_inference(buttom_img,thresholds)
#            img[middle_point:] = buttom_predict_img
            blurred_img[middle_point:] = buttom_predict_img
        
        base64img = self._imagetobase64(blurred_img) 

        return {"image": base64img}

    def crop(self,thresholds:float,percent_crop:float=0.0) ->Dict[str,ByteString]:

        img = self.base64toimage
        blurred_img = cv.GaussianBlur(img, (21, 21), 0) 

        if percent_crop > 0.0:
            height,width,_ = img.shape
            cut_pixel_height_side = int(percent_crop*height)//2
            cut_pixel_width_side = int(percent_crop*width)//2

            crop_img = img[cut_pixel_height_side : -cut_pixel_height_side , cut_pixel_width_side : -cut_pixel_width_side]
            crop_predict_img = self.run_inference(crop_img,thresholds)
#            img[cut_pixel_height_side : -cut_pixel_height_side , cut_pixel_width_side : -cut_pixel_width_side] = crop_predict_img 
            blurred_img[cut_pixel_height_side : -cut_pixel_height_side , cut_pixel_width_side : -cut_pixel_width_side] = crop_predict_img

        else:
            normal_img = img
            img = self.run_inference(normal_img,thresholds)
            blurred_img = img  # when removing blur logic remove it
        
        base64img = self._imagetobase64(blurred_img)
        return {"image": base64img}

    def amount_cut_images(self,x_axis:Tuple[float,float]=None,y_axis:Tuple[float,float]=None) -> Dict[str,ByteString]:
        
        img = self.base64toimage
        blurred_img = cv.GaussianBlur(img, (21, 21), 0) 
        
        if isinstance(x_axis,Tuple) or isinstance(y_axis,Tuple):
            height,width,_ = img.shape

            if isinstance(x_axis,tuple):
                start_pixel_width = int(width*x_axis[0])
                end_pixel_width = int(width*x_axis[1])
                cut_img_width = img[ : ,   start_pixel_width: end_pixel_width]

            if isinstance(y_axis,tuple):
                start_pixel_height = int(height*y_axis[0])
                end_pixel_height = int(height*y_axis[1])

                if isinstance(x_axis,tuple):
                    cut_img_height_width =   cut_img_width[start_pixel_height: end_pixel_height]
                    cut_img = self.run_inference(cut_img_height_width,0.5)
                else:
                    cut_img_height = img[  start_pixel_height:   end_pixel_height]
                    # cv.imwrite('width.jpg', cut_img_height)
                    cut_img = self.run_inference(cut_img_height,0.5)

            elif isinstance(x_axis,tuple):
                cut_img = self.run_inference(cut_img_width,0.5)
                
            # cv.imwrite("cut.jpg",cut_img)

            if isinstance(y_axis,tuple):
                if isinstance(x_axis,tuple):
                    cut_img_width[start_pixel_height: end_pixel_height] = cut_img
#                    img[ : ,   start_pixel_width: end_pixel_width] = cut_img_width
                    blurred_img[ : ,   start_pixel_width: end_pixel_width] = cut_img_width
                else:
#                    img[  start_pixel_height:   end_pixel_height] = cut_img
                    blurred_img[  start_pixel_height:   end_pixel_height] = cut_img
            elif isinstance(x_axis,tuple):
#                img[ : ,   start_pixel_width: end_pixel_width] = cut_img
                blurred_img[ : ,   start_pixel_width: end_pixel_width] = cut_img

        else:
            normal_img = img
            img = self.run_inference(normal_img,0.5)
            blurred_img = img
        
        base64img = self._imagetobase64(blurred_img)

        return {"image": base64img}



