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
        print("Overriding tract/patch info and looking for HSC file at requested coordinates.")
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


def get_hsc_DR3_data(dirpath, filters=['g','r','i'], tract=10054, patch=[0,0], coord=None, cutout_size=[128, 128]):
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
        print("Overriding tract/patch info and looking for HSC file at requested coordinates.")
        tract, patch = get_tract_patch_from_coord(coord)
        
    datas = []

    for f in filters:
        filepath = os.path.join(dirpath, f'HSC-{f}/calexp-HSC-{f}-{tract}-{patch[0]},{patch[1]}.fits')
        
        print(f'Loading "{filepath}".')
        #try:
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
        #except:
        #    print('Missing filter ', f)
            

        return np.array(datas)
    
    
    

def read_image(filenames, normalize='lupton', stretch=5, Q=10, m=0, ceil_percentile=99.995, dtype=np.uint8, A=1e4):
    # Read image
    
    g = fits.getdata(os.path.join(filenames[0]), memmap=False)
    r = fits.getdata(os.path.join(filenames[1]), memmap=False)
    z = fits.getdata(os.path.join(filenames[2]), memmap=False)
    
    # Contrast scaling / normalization
    I = (z + r + g)/3.0
    
    length, width = g.shape
    image = np.empty([length, width, 3], dtype=dtype)
    
    # Options for contrast scaling
    if normalize.lower() == 'lupton' or normalize.lower() == 'luptonhc':
        z = z*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        r = r*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        g = g*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        
    elif normalize.lower() == 'zscore':
        
        Isigma = I*np.mean([np.nanstd(g), np.nanstd(r), np.nanstd(z)])
        z = (z - np.nanmean(z) - m)/(Isigma+1e-5)
        r = (r - np.nanmean(r) - m)/(Isigma+1e-5)
        g = (g - np.nanmean(g) - m)/(Isigma+1e-5)
        
        #zsigma = np.nanstd(z)
        #rsigma = np.nanstd(r)
        #gsigma = np.nanstd(g)
        
        #z = A*(z - np.nanmean(z) - m)/zsigma
        #r = A*(r - np.nanmean(r) - m)/rsigma
        #g = A*(g - np.nanmean(g) - m)/gsigma
        
        
    elif normalize.lower() == 'zscore_orig':
        
        zsigma = np.nanstd(z)
        rsigma = np.nanstd(r)
        gsigma = np.nanstd(g)
        
        z = A*(z - np.nanmean(z) - m)/zsigma
        r = A*(r - np.nanmean(r) - m)/rsigma
        g = A*(g - np.nanmean(g) - m)/gsigma

        image[:,:,0] = z # R
        image[:,:,1] = r # G
        image[:,:,2] = g # B
        
        return image
        
    elif normalize.lower() == 'sinh':
        z = np.sinh((z-m))
        r = np.sinh((r-m))
        g = np.sinh((g-m))

        
        
    elif normalize.lower() == 'sqrt':
        z = np.sqrt((z-m))
        r = np.sqrt((r-m))
        g = np.sqrt((g-m))
        
    elif normalize.lower() == 'linear':
        z = (z - m)/I
        r = (r - m)/I
        g = (g - m)/I
        
    elif normalize.lower()== 'raw':
        image = np.zeros([length, width, 3])
        image[:,:,0] = g # R
        image[:,:,1] = r # G
        image[:,:,2] = z # B
        return image
        
    
    else:
        print('Normalize keyword not recognized.')

    max_RGB = np.nanpercentile([z, r, g], ceil_percentile)
    #max_RGB = np.nanpercentile([z, r, g], ceil_percentile)

    z = np.clip(z,None,max_RGB)
    r = np.clip(r,None,max_RGB)
    g = np.clip(g,None,max_RGB)
    
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
    
    if normalize.lower() == 'luptonhc':
        factor = 2 #gives original image
        cenhancer = ImageEnhance.Contrast(Image.fromarray(image))
        im_output = cenhancer.enhance(factor)
        benhancer = ImageEnhance.Brightness(im_output)
        image = benhancer.enhance(factor)
    
    return np.asarray(image)