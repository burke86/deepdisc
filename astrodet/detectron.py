import sys, os
import numpy as np
import json
import cv2
from astropy.io import fits


from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode
#Yufeng Dec21 more import
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

#Yufeng Dec 21 add astro_metadata into parameters
def plot_stretch_Q(dataset_dicts, astro_metadata, num=0, stretches=[0.01,0.1,0.5,1], Qs=[1,10,5,100], ceil_percentile=99.5):
    """
    Plots different normalizations of your image using the stretch, Q parameters. 
    
    Parameters
    ----------
    dataset_dicts : dict
        detectron dataset dictionary
    num : int
        Dataset number/index to use
    stretches : array
        List of stretch params you want to permutate through to find optimal image normalization.
        Default is [0.01, 0.1, 0.5, 1]
    Qs : array
        List of Q params you want to permutate through to find optimal image normalization.
        Default is [1, 10, 5, 100]
        
    Code adapted from:
        https://pmelchior.github.io/scarlet/tutorials/display.html
        
    Returns
    -------
    fig : Figure object
    
    """
    
    d = dataset_dicts[num]
    
    fig, ax = plt.subplots(len(stretches), len(Qs), figsize=(9,9))
    for i, stretch in enumerate(stretches):
        for j, Q in enumerate(Qs):
            img = read_image(d, normalize="lupton", stretch=stretch, Q=Q, ceil_percentile=ceil_percentile)
            # Scale the RGB channels for the image
            visualizer = Visualizer(img, metadata=astro_metadata)
            out = visualizer.draw_dataset_dict(d)
            ax[i][j].imshow(out.get_image(), origin='lower')
            ax[i][j].set_title("Stretch {}, Q {}".format(stretch, Q), fontsize=10)
            ax[i][j].axis('off')
            
    return fig


#Yufeng dec 23 dtype should be set to uint8, uint16 does not work
def read_image(dataset_dict, normalize='lupton', stretch=5, Q=10, m=0, ceil_percentile=99.995, dtype=np.uint8):
    # Read image
    
    g = fits.getdata(os.path.join(dataset_dict['filename_G']), memmap=False)
    r = fits.getdata(os.path.join(dataset_dict['filename_R']), memmap=False)
    z = fits.getdata(os.path.join(dataset_dict['filename_I']), memmap=False)
    
    # Contrast scaling / normalization
    I = (z + r + g)/3.0
    
    length, width = g.shape
    image = np.empty([length, width, 3], dtype=dtype)
    
    # Options for contrast scaling
    if normalize.lower() == 'lupton':
        z = z*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        r = r*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        g = g*np.arcsinh(stretch*Q*(I - m))/(Q*I)
    elif normalize.lower() == 'zscore':
        Isigma = I*np.mean([np.nanstd(g), np.nanstd(r), np.nanstd(z)])
        z = (z - np.nanmean(z) - m)/Isigma
        r = (r - np.nanmean(r) - m)/Isigma
        g = (g - np.nanmean(g) - m)/Isigma
    elif normalize.lower() == 'linear':
        z = (z - m)/I
        r = (r - m)/I
        g = (g - m)/I
    else:
        print('Normalize keyword not recognized.')

    max_RGB = np.nanpercentile([z, r, g], ceil_percentile)
    # avoid saturation
    r = r/max_RGB; g = g/max_RGB; z = z/max_RGB

    # Rescale to 0-255 for dtype=np.uint8
    max_dtype = np.iinfo(dtype).max
    r = r*max_dtype
    g = g*max_dtype
    z = z*max_dtype

    # 0-255 RGB image
    image[:,:,0] = z # R
    image[:,:,1] = r # G
    image[:,:,2] = g # B
    
    return image