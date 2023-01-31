import sys, os
import numpy as np
import json
import cv2
from astropy.io import fits

import detectron2.data as data
import detectron2.data.transforms as T
from detectron2.data.transforms import Transform
from detectron2.data.transforms import Augmentation


from detectron2.data.transforms import Transform
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, TransformList

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




#Code taken from Deshwal on Stack Overflow

class GenericWrapperTransform(Transform):
    """
    Generic wrapper for any transform (for color transform only. You can give functionality to apply_coods, apply_segmentation too)
    """

    def __init__(self, custom_function):
        """
        Args:
            custom_function (Callable): operation to be applied to the image which takes in an ndarray and returns an ndarray.
        """
        if not callable(custom_function):
            raise ValueError("'custom_function' should be callable")
        
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        '''
        apply transformation to image array based on the `custom_function`
        '''
        return self.custom_function(img)

    def apply_coords(self, coords):
        '''
        Apply transformations to Bounding Box Coordinates. Currently is won't do anything but we can change this based on our use case
        '''
        return coords

    def inverse(self):
        return T.NoOpTransform()

    def apply_segmentation(self, segmentation):
        '''
        Apply transformations to segmentation. currently is won't do anything but we can change this based on our use case
        '''
        return segmentation


class CustomAug(Augmentation):
    """
    Given a probability and a custom function, return a GenericWrapperTransform object whose `apply_image`  
    will be called to perform augmentation
    """

    def __init__(self, custom_function, prob=1.0):
        """
        Args:
            custom_op: Operation to use. Must be a function takes an ndarray and returns an ndarray
            prob (float): probability of applying the function
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        '''
        Based on probability, choose whether you want to apply the given function or not
        '''
        do = self._rand_range() < self.prob
        if do:
            return GenericWrapperTransform(self.custom_function)
        else:
            return T.NoOpTransform() # it returns a Transform which just returns the original Image array only


class KRandomAugmentationList(Augmentation):
    """
    Select and Apply "K" augmentations in "RANDOM" order with "Every"  __call__ method invoke
    """
    def __init__(self, augs, k:int = -1, cropaug=None):
        """
        Args:
            augs: list of [Augmentation or Transform]
            k: Number of augment to use from the given list in range [1,len_augs].
            If None, use all. If it is -1, generate K randomly between [1,len_augs]
        """
        super().__init__()
        self.max_range = len(augs)
        self.k = k
        self.augs = augs # set augs to use as fixed if we have to use same augs everytime
        self.cropaug = cropaug

    def _setup_augs(self, augs, k:int):
        '''
        Setup the argument list. Generates the list of argument to use from the given list
        args:
            augs: list of [Augmentation or Transform])
            k: Number of augment to use from the given list in range [1,len_augs].
            If False, use all. If it is -1, generate K randomly between [1,len_augs]
        '''
        if k == -1: # Generate a random number
            k = np.random.randint(1,len(augs)+1)

        elif k is None: # use all
            k = self.max_range

        temp = np.random.choice(augs,k,replace=False) # get k augments randomly
        augl=[_transform_to_aug(x) for x in temp]
        if self.cropaug is None:
            pass
        else:
            augl.insert(0,self.cropaug)

        return augl


    def __call__(self, aug_input) -> Transform:
        tfms = []

        for x in self._setup_augs(self.augs, self.k): # generate auguments to use randomly on the fly
            #print(x)
            tfm = x(aug_input)
            tfms.append(tfm)
        return TransformList(tfms)

    def __repr__(self):
        msgs = [str(x) for x in self.augs]
        return "AugmentationList[{}]".format(", ".join(msgs))

    __str__ = __repr__

## Added for importing
class _TransformToAug(Augmentation):
    def __init__(self, tfm: Transform):
        self.tfm = tfm

    def get_transform(self, *args):
        return self.tfm

    def __repr__(self):
        return repr(self.tfm)

    __str__ = __repr__


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:
        return _TransformToAug(tfm_or_aug)

        

