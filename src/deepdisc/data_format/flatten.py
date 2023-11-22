
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import deepdisc

PATH = deepdisc.__path__[0]

def flatten_dc2(ddicts):
    i=0
    images=[]
    metadatas = []
    for d in ddicts:
        filename= d[f"filename"]
        for a in d['annotations']:
            new_dict = {}
            new_dict["image_id"] = 1
            new_dict["height"] = 128
            new_dict["width"] = 128

            x = a['bbox'][0]
            y = a['bbox'][1]
            w = a['bbox'][2]
            h = a['bbox'][3]

            xnew = x+w//2-64
            ynew = y+h//2-64

            if xnew<0 or ynew <0 or xnew+128>d['height'] or ynew+128>d['height'] or a['mag_i']>25.3:
                continue

            bxnew = x-(x+w//2 - 64)
            bynew = y-(y+h//2 - 64)
            print(filename)
            #base=filename.split('.')[0].split('/')[-1]
            #dirpath = '/home/g4merz/DC2/nersc_data/scarlet_data'
            #fn=os.path.join(dirpath,base)+'.npy'
            
            
            base=os.path.join(os.path.dirname(os.path.dirname(PATH)),filename.split('.fits')[0])
            print(base)
            fn = base+'.npy'
            
            
            image = np.load(fn)

            imagecut = image[:,ynew:ynew+128,xnew:xnew+128]
            #imagecut = image[:,xnew:xnew+128,ynew:ynew+128]

            #imagecut=imagecut.reshape(imagecut.shape[0],-1)
            images.append(imagecut.flatten())

            metadata =[128,128,i,bxnew,bynew,w,h,1,a['category_id'],a['redshift'],a['obj_id'],a['mag_i']]
            metadatas.append(metadata)
            i+=1
            
    images = np.array(images)
    metadatas = np.array(metadatas)
    
    flattened_data = []
    for image,metadata in zip(images,metadatas):
        #flatdat = np.concatenate((image,metadat.iloc[i].values))
        flatdat = np.concatenate((image,metadata))
        flattened_data.append(flatdat)

            
    return flattened_data
                    





