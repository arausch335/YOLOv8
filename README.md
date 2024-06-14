# YOLO v8 Auto-Annotation Pipeline

This tool provides an efficient pipeline to speed up the segmentation annotation process in CVAT 
by training a YOLOv8 model on available annotations and using it to predict new annotations.
It converts a folder of images and a COCO-JSON file into the YOLO dataset format.

## Instructions
- Annotate some images in CVAT and export the COCO JSON format, saving it under original annotations
- Import the images into the images directory, making sure they have the same name as in the annotations
- Download weights from Ultralytics if not already present
- Create a Model object in handler.py
- Run Model.train, Model.auto_annotate, Model.predict
- Auto annotations will appear in the data directory under auto_annotations