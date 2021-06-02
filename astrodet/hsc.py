import sys, os
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS

import matplotlib
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.nddata import Cutout2D

def get_tract_patch_from_coord(coord, f='data/hsc/tracts_patches_UD-COSMOS.txt'):
    """
    Get tract and patch from SkyCoord
    
    Parameters
    ----------
    coord : SkyCoord
        Coodinate of object
    f     : str
        Path to UD-COSMOS.txt data file
    
    Returns
    -------
    tract : Astropy HDUData
        Tract ID
    patch : [int, int]
        Patch ID tuple
    """

    lines = open(f).read().splitlines()
    # Keep only field centers
    centers = [l for l in lines if "Center (RA, Dec):" in l and 'Patch:' in l]
    # Extract tracts and patches at each line
    tracts = [int(c.split('Tract: ')[1].split('Patch: ')[0]) for c in centers]
    patches = [c.split('Patch: ')[1].split('Center ')[0] for c in centers]
    patches = [tuple(map(int, p.split(','))) for p in patches]
    # Extract coord of field centers
    center_coords = [c.split('Center (RA, Dec): ')[1] for c in centers]
    ra = [float(c.replace('(', '').split(',')[0]) for c in center_coords]
    dec = [float(c.replace(')', '').split(',')[1]) for c in center_coords]
    # Convert to SkyCoord
    center_coords = SkyCoord(ra, dec, unit='deg')
    # Find closest field center to requested coordinate
    idx, d2d, d3d = match_coordinates_sky(coord, center_coords)
    # No sep contraint
    tract = tracts[idx]
    patch = patches[idx]
    
    return tract, patch

def get_hsc_data(dirpath, filters=['g','r','i'], tract=10054, patch=[0,0], coord=None, cutout_size=[128, 128]):
    """
    Get HSC data given tract/patch info or SkyCoord
    
    Parameters
    ----------
    dirpath : str
        Path to HSC image file directory
    filters : list 
        A list of filters for your images. Default is ['g', 'r', 'i'].
    tract  : int
        An integer used for specifying the tract. Default is 10054
    patch : [int, int]
        Patch #,#. Default is [0,0]
    coord  : SkyCoord
        Astropy SkyCoord, when specified, overrides tract/patch info and attempts to lookup HSC filename from ra, dec. 
        Default is None
    cutout_size: [int, int]
        Size of cutout to use (set to None for no cutting). Default is [128, 128]
        
    The image filepath is in the form:
        {dirpath}/deepCoadd/HSC-{filter}/{tract}/{patch[0]},{patch[1]}/calexp-HSC-{filter}-{tract}-{patch[0]},{patch[1]}.fits
    
    Returns
    -------
    data : ndarray
        HSC data array with dimensions [filters, N, N]
    """
    
    filters = [f.upper() for f in filters]
    
    if coord is not None:
        import warnings
        warnings.warn("Overriding tract/patch info and looking for HSC file at requested coordinates.")
        tract, patch = get_tract_patch_from_coord(coord)
        
    datas = []

    for f in filters:
        filepath = os.path.join(dirpath, f'HSC-{f}/{tract}/{patch[0]},{patch[1]}/calexp-HSC-{f}-{tract}-{patch[0]},{patch[1]}.fits')
        
        print(f'Loading "{filepath}".')
        
        obs_hdul = fits.open(filepath)
        data = obs_hdul[1].data
        wcs = WCS(obs_hdul[1].header)
        
        # Cutout data at center of patch (coord=None) or at coord (if specified)
        if cutout_size is not None:
            # Use coord for center position if specified
            if coord is None:
                shape = np.shape(data)
                position = (shape[0]/2, shape[1]/2)
            else:
                position = coord
            data = Cutout2D(data, position=position, size=cutout_size, wcs=wcs).data
            
        datas.append(data)
    
    return np.array(datas)