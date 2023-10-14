import os
import os.path
import pandas as pd
from PIL import Image

def get_dataframe_annotations_class_string(ann_path, img_path, dict):
    annos = []
    # Read txts  
    for files in os.walk(ann_path):
        for file in files[2]:
            # Read image and get its size attributes
            img_name = os.path.splitext(file)[0] + '.jpg'
            fileimgpath = os.path.join(img_path ,img_name)
            im = Image.open(fileimgpath)
            w = int(im.size[0])
            h = int(im.size[1])

            # Read txt file 
            filelabel = open(os.path.join(ann_path , file), "r")
            lines = filelabel.read().split('\n')
            obj = lines[:len(lines)-1]  
            for i in range(0, int(len(obj))):
                objbud=obj[i].split(' ')                
                name = dict[objbud[0]]
                x1 = float(objbud[1])
                y1 = float(objbud[2])
                w1 = float(objbud[3])
                h1 = float(objbud[4])

                xmin = int((x1*w) - (w1*w)/2.0)
                ymin = int((y1*h) - (h1*h)/2.0)
                xmax = int((x1*w) + (w1*w)/2.0)
                ymax = int((y1*h) + (h1*h)/2.0)

                annos.append([img_name, w ,h ,name ,xmin ,ymin ,xmax ,ymax])
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax' ]
    df = pd.DataFrame(annos, columns=column_name)        
    return df

def get_dataframe_annotations(ann_path, img_path):
    annos = []
    # Read txts  
    for files in os.walk(ann_path):
        for file in files[2]:
            # Read image and get its size attributes
            img_name = os.path.splitext(file)[0] + '.jpg'
            fileimgpath = os.path.join(img_path ,img_name)
            im = Image.open(fileimgpath)
            w = int(im.size[0])
            h = int(im.size[1])

            # Read txt file 
            filelabel = open(os.path.join(ann_path , file), "r")
            lines = filelabel.read().split('\n')
            obj = lines[:len(lines)-1]  
            for i in range(0, int(len(obj))):
                objbud=obj[i].split(' ')                
                class_id = objbud[0]
                x1 = float(objbud[1])
                y1 = float(objbud[2])
                w1 = float(objbud[3])
                h1 = float(objbud[4])

                xmin = int((x1*w) - (w1*w)/2.0)
                ymin = int((y1*h) - (h1*h)/2.0)
                xmax = int((x1*w) + (w1*w)/2.0)
                ymax = int((y1*h) + (h1*h)/2.0)

                annos.append([img_name, w, h , class_id, xmin ,ymin, xmax ,ymax])
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax' ]
    df = pd.DataFrame(annos, columns=column_name)        
    return df