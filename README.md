# Detectron2

Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook.


# BrandRecognition

A brand logo detection system using Detectron2 API. This API will detect the type of logo in the image and this API is also compatible to Accept the Region of interest(ROI) of the detection object 

Acceptable ROI :  ['TOP','BUTTOM','LEFT',RIGHT','CENTER CROP']

## Dataset for Brand_recognition

 Download the flickr logos 27 dataset from [here](http://image.ntua.gr/iva/datasets/flickr_logos/).

   The flickr logos 27 dataset contains 27 classes of brand logo images downloaded from Flickr. The brands included in the dataset are: Adidas, Apple, BMW, Citroen, Coca Cola, DHL, Fedex, Ferrari, Ford, Google, Heineken, HP, McDonalds, Mini, Nbc, Nike, Pepsi, Porsche, Puma, Red Bull, Sprite, Starbucks, Intel, Texaco, Unisef, Vodafone and Yahoo.

   ```shell
   $ wget http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz
   $ tar zxvf flickr_logos_27_dataset.tar.gz
   $ cd flickr_logos_27_dataset
   $ tar zxvf flickr_logos_27_dataset_images.tar.gz
   $ cd ../
   ```

# Preprocessing steps done

1) There is no XML file attached, so you have to generate the XML file from the annotation text file.
2) Refer the Ipython notebooks in the Notebook folder for reference . ( will add .py file sooner)

# How to use the Brand_Recognition FASTAPI in local

   ```bash
   $ git clone https://github.com/kkkumar2/Brand_Recognition.git
   ```
   ```python
   uvicorn app:app --reload
   ``` 
# Glimpse of how Fastapi with swagger ui will look

![example1](detections\api.PNG)

# Original sample image

|![example1](detections\input.jpg)|

# Prediction examples (Full image detection)

|![example1](detections\full_prediction.jpg)|

# Prediction examples (ROI based image detection LEFT)

|![example1](detections\left_prediction.jpg)|

# Prediction examples (ROI based image detection RIGHT)

|![example1](detections\right_prediction.jpg)|

# deployment

Need to deploy the API