# AITableGameMonitor
AI based Table Game Monitoring 



# Architecture and workflow

For main monitoring program 

1. geometry transform 
2. forground detection 
3. object detection and tracking (YOLO ad SORT)
4. classification (mobilenet) 
5. baccarat rules
6. custom algorithms 


```

 input frame         
     |               layout.jpg
     v                  | 
[alignment]             |
     | aligned frame    |
     v                  |
    YOLO                |
     | cls,bbox,conf  fg det  
     v                  |
  (SORT)                |
     | id,cls,bbox      | 
     v                  | layout.json
 loca event checker <---+         
     |
     v
  classifier            
     | card cls, chip cls
     v
   events

```

# prerequisite

For each new table configuraiton: 

1. the layout should be extracted (manually) to tablelayout.json using tablelayout.py 
2. the first clean table image should be extracted to tablelayout.jpg 


# how to run 

## model files and test video 

 - test video : https://drive.google.com/file/d/1kcsukjzLj24-nv33NljRZyDE0SCqVAHi/view?usp=sharing
 - retrained YOLO: https://drive.google.com/file/d/1iPZwNRoWBz_XR0Q07kKH1AZ77X9tR1Wu/view?usp=sharing
 - retained Mobilenet: https://drive.google.com/file/d/1fNt76VW8V9wno1YuarNTNacZYq_jYJlC/view?usp=sharing

## the main baccarat monitoring  

 - use table layout info (json file) 
 - use clean table image for background substraction (optional)
 - use YOLO to detect hands, chips, and cards
 - statemachine for baccarat games   


```
python  main.py  <layout.jpg> <baccarat_game_video>

```

## layout generation 

 - provide manual drawing layout (@TODO: automatic)
 - create or edit layout json file for the clear table image 

```
python  tablelayout.py <template image> 
```

## train yolo for table game 

training, exporting, image testing and tracking video 

### training a yolo model on new dataset (yaml file)
```
python  yolo_tablegame.py  <yolo_tablegame>.yaml 
```
The result Pytorch mode file will be saved in run/*/weights/best.pt 
you have to copy it to yolo11n_tablegame.pt

### RKNN 

  convert Pytorch to RKNN, and test  (on Slot eye 2) 

```
python yolo_rknn.py <trained yolo>.pt 
python yolo_rknn.py <trained yolo>.onnx 
python yolo_rknn.py <trained yolo>.rknn <testimg> 

```

### test object tracking for a video  

```
python  yolo_tablegame.py <table_game_video>.mp4 

```
### test object detection for an image  

```
python  yolo_tablegame.py  <table_image>.jpg 

```

##  data processing  

 detect hands and faces, and mosaicing or generatiing YOLO labels and images 

```
mosaic.py <videofile>.mp4

```

## Card Classification 

Mobilenet is trained for card classification (the training data from the Kaggle site, cen be shared)


### Training 


 you need to make your own dataset for your applicaiton.

 The sample dataset can be downloaded here: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

```
python mmobilenet_card_classifier.py train 
```


### Test  
```
python mmobilenet_card_classifier.py test  <direcotry/imagefile> 
```

### API

API is defined and used in main.py and yolo_tabelgame.py. 



### Info  
```
python mmobilenet_card_classifier.py check 
```




