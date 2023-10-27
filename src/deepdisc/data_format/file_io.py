import json

import numpy as np
from astropy.visualization import make_lupton_rgb


def get_data_from_json(file):
    """Open a JSON text file, and return encoded data as dictionary.

    Parameters
    ----------
    file : str
        pointer to file

    Returns
    -------
        dictionary of encoded data
    """
    # Opening JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class ImageReader:
    """Class that will read images on the fly for the training/testing dataloaders"""

    def raw(im):
        return im.astype(np.float32)

    def lupton(im, bandlist=[2, 1, 0], stretch=0.5, Q=10, m=0):
        """
        Parameters
        ----------
        im : np array
            The image being scaled
        bandlist : list[int]
            Which bands to use for lupton scaling (must be 3)
        stretch : float
            lupton stretch parameter
        Q : float
            lupton Q parameter
        m: float
            lupton minimum parameter

        """

        assert np.array(im.shape).argmin() == 2 and len(bandlist) == 3
        b1 = im[:, :, bandlist[0]]
        b2 = im[:, :, bandlist[1]]
        b3 = im[:, :, bandlist[2]]

        image = make_lupton_rgb(b1, b2, b3, minimum=m, stretch=stretch, Q=Q)
        return image

    def zscore(im, A=1):
        """
        Parameters
        ----------
        im : np array
            The image being scaled
        A : float
            A multiplicative scaling factor applied to each band

        """

        I = np.mean(im, axis=-1)
        Imean = np.nanmean(I)
        Isigma = np.nanstd(I)

        for i in range(im.shape[-1]):
            image[:, :, i] = A * (im[:, :, i] - Imean - m) / Isigma

        return image

    norm_dict = {"raw": raw, "lupton": lupton}

    def __init__(self, reader, norm="raw", **scalekwargs):
        """
        Parameters
        ----------
        reader : function
            This function should take a single key and return a single image as a numpy array
            ex) give a filename or an index in an array
        norm : str
            A contrast scaling to apply before data augmentation, i.e. luptonizing or z-score scaling
            Default = raw
        **scalekwargs : key word args
            Key word args for the contrast scaling function

        """

        self.reader = reader

        self.scalekwargs = scalekwargs
        self.scaling = ImageReader.norm_dict[norm]

    def __call__(self, key):
        im = self.reader(key)
        im_scale = self.scaling(im, **self.scalekwargs)
        return im_scale

    @classmethod
    def add_scaling(cls, name, func):
        """Add a custom contrast scaling function

        ex)
        def sqrt(image):
            image[:,:,0] = np.sqrt(image[:,:,0])
            image[:,:,1] = np.sqrt(image[:,:,1])
            image[:,:,2] = np.sqrt(image[:,:,2])
            return image

        ImageReader.add_scaling('sqrt',sqrt)


        """
        cls.norm_dict[name] = func
