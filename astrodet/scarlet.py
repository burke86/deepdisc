import sys, os
import numpy as np

import scarlet
import sep

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.wcs import WCS


def plot_stretch_Q(datas, stretches=[0.01,0.1,0.5,1], Qs=[1,10,5,100]):
    """
    Plots different normalizations of your image using the stretch, Q parameters. 
    
    Parameters
    ----------
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
    
    fig, ax = plt.subplots(len(stretches), len(Qs), figsize=(9,9))
    for i, stretch in enumerate(stretches):
        for j, Q in enumerate(Qs):
            asinh = scarlet.display.AsinhMapping(minimum=0, stretch=stretch, Q=Q)
            # Scale the RGB channels for the image
            img_rgb = scarlet.display.img_to_rgb(datas, norm=asinh)
            ax[i][j].imshow(img_rgb)
            ax[i][j].set_title("Stretch {}, Q {}".format(stretch, Q))
            ax[i][j].axis('off')
            
    return fig


def make_catalog(datas, lvl=4, wave=True, subtract_background=False, segmentation_map=False, maskthresh=5.0, object_limit=100000):
    """
    Creates a detection catalog by combining low and high resolution data
    
    Parameters
    ----------
    datas: array
        array of Data objects
    lvl: int
        detection lvl
    wave: Bool
        set to True to use wavelet decomposition of images before combination
    subtract_background : Bool
        if you want to subtract the background and retrieve an estimate, change to True. But default 
        is False because HSC images are already background subtracted.
    segmentation_map : Bool
        Whether to run sep segmentation map
    maskthresh : bool
        Mask threshold for sep segmenation
    object_limit : int
        Limit on number of objects to detect in image
        
    Code adapted from https://pmelchior.github.io/scarlet/tutorials/wavelet_model.html
    
    Returns
    -------
    catalog: sextractor catalog
        catalog of detected sources (use 'catalog.dtype.names' for info)
    bg_rms: array
        background level for each data set (set to None if subtract_background is False)
    """
        
    if type(datas) is np.ndarray:
        hr_images = datas / np.sum(datas, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(hr_images, axis=0)
    else:
        data_lr, data_hr = datas
        # Create observations for each image
        # Interpolate low resolution to high resolution
        interp = interpolate(data_lr, data_hr)
        # Normalization of the interpolate low res images
        interp = interp / np.sum(interp, axis=(1, 2))[:, None, None]
        # Normalisation of the high res data
        hr_images = data_hr.images / np.sum(data_hr.images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(interp, axis=0) + np.sum(hr_images, axis=0)
        detect_image *= np.sum(data_hr.images)
    if np.size(detect_image.shape) == 4:
        if wave:
            # Wavelet detection in the first three levels
            wave_detect = Starlet(detect_image.mean(axis=0), lvl=5).coefficients
            wave_detect[:, -1, :, :] = 0
            detect = scarlet.Starlet(coefficients=wave_detect).image
        else:
            # Direct detection
            detect = detect_image.mean(axis=0)
    else:
        if wave:
            wave_detect = scarlet.Starlet(detect_image).coefficients
            detect = wave_detect[0][0] + wave_detect[0][1] + wave_detect[0][2]
        else:
            detect = detect_image
    
    bkg = sep.Background(detect)
    # Set the limit on the number of sub-objects when deblending. Default is 1024
    sep.set_sub_object_limit(object_limit)
    
    # Extract detection catalog with segmentation maps!
    # Can use this to retrieve ellipse params
    catalog = sep.extract(detect, lvl, err=bkg.globalrms, segmentation_map=segmentation_map, maskthresh=maskthresh)
        
    # Estimate background
    if subtract_background == True:
        if type(datas) is np.ndarray:
            bkg_rms = scarlet.wavelet.mad_wavelet(datas)

        else:
            bkg_rms = []
            for data in datas:
                bkg_rms.append(scarlet.wavelet.mad_wavelet(data.images))
                
    else:
        bkg_rms = None

    return catalog, bkg_rms


def fit_scarlet_blend(starlet_sources, observation, max_iters=200, e_rel=1e-4, plot_likelihood=True):
    """
    Creates a detection catalog by combining low and high resolution data
    
    Parameters
    ----------
    datas: array
        array of Data objects
        
        Will end early if likelihood and constraints converge
    
    Returns
    -------
    """
    # Create and fit Blend model. Go for 200 iterations,
    # but will end early if likelihood and constraints converge
    print(f"Fitting Blend model.")
    try:
        starlet_blend = scarlet.Blend(starlet_sources, observation)
        it, logL = starlet_blend.fit(max_iters, e_rel=e_rel)
        print(f"Scarlet ran for {it} iterations to logL = {logL}")

    # Catch any exceptions like no detections
    except AssertionError as e1:
        print(f"Length of detection catalog is {len(catalog)}.")

    if plot_likelihood == True:
        scarlet.display.show_likelihood(starlet_blend)
        plt.show()
    
    return starlet_blend, logL


def _plot_wavelet(datas):
    """
    Helper function to plot wavelet transformation diagnostic figures with scarlet
    
    Parameters
    ----------
    datas: array
        array of Data objects
    
    Returns
    -------
    """
    
    # Declare a starlet object (and performs the transform)
    Sw = scarlet.Starlet(datas, lvl=5, direct=True)
    # This is the starlet transform as an array
    w = Sw.coefficients
    # The inverse starlet transform of w (new object otherwise, the tranform is not used)
    iw = Sw.image

    # TODO: Clean this code up using plt.subplots()
    # The wavelet transform of the first slice of images in pictures
    lvl = w.shape[1]
    plt.figure(figsize=(lvl*5+5,5))
    plt.suptitle('Wavelet coefficients')
    for i in range(lvl):
        plt.subplot(1, lvl, i+1)
        plt.title('scale' + str(i+1))
        plt.imshow(w[0,i], cmap='inferno')
        plt.colorbar()
    plt.show()

    # Making sure we recover the original image
    plt.figure(figsize=(30,10))
    plt.subplot(131)
    plt.title('Original image', fontsize=20)
    plt.imshow(datas[0], cmap='inferno')
    plt.colorbar()
    plt.subplot(132)
    plt.title('Starlet-reconstructed image', fontsize=20)
    plt.imshow(iw[0], cmap='inferno')
    plt.colorbar()
    plt.subplot(133)
    plt.title('Absolute difference', fontsize=20)
    plt.imshow((np.abs(iw[0]-datas[0])), cmap='inferno')
    plt.colorbar()
    plt.show()
    
    return


def _plot_scene(starlet_sources, observation, norm, catalog, show_model=False, show_rendered=True,
               show_observed=True, show_residual=True, add_labels=True, add_boxes=False,
               add_ellipses=True):
    
    """
    Helper function to plot scene with scarlet
    
    Parameters
    ----------
    starlet_sources: List
        List of ScarletSource objects
    observation: 
        Scarlet observation objects
    norm:
        Scarlet normalization for plotting
    catalog: list
        Source detection catalog
    show_model: bool
        Whether to show model
    show_rendered: bool
        Whether to show rendered model
    show_observed: bool
        Whether to show observed
    show_residual: bool
        Whether to show residual
    add_labels: bool
        Whether to add labels
    add_boxes: bool
        Whether to add bounding boxes to each panel
    add_ellipses: bool
        Whether to add ellipses to each panel
    
    
    Returns
    -------
    fig : matplotlib Figure
        Figure object
    """
    
    fig = scarlet.display.show_scene(starlet_sources, observation=observation, norm=norm,
                                     show_model=show_model, show_rendered=show_rendered,
                                     show_observed=show_observed, show_residual=show_residual,
                                     add_labels=add_labels, add_boxes=add_boxes)
        
    for ax in fig.axes:

        # Plot sep ellipse around all sources from the detection catalog
        if add_ellipses == True:
            for k, src in enumerate(catalog):

                # See https://sextractor.readthedocs.io/en/latest/Position.html
                e = Ellipse(xy=(src['x'], src['y']),
                            width=6*src['a'],
                            height=6*src['b'],
                            angle=np.rad2deg(src['theta']))

                e.set_facecolor('none')
                e.set_edgecolor('white')

                ax.add_artist(e)

        ax.axis('off')

    fig.subplots_adjust(wspace=0.01)
    plt.show()
    
    return fig

def run_scarlet(datas, filters, stretch=0.1, Q=5, sigma_model=1, sigma_obs=5,
                subtract_background=False, max_chi2=5000, morph_thresh=0.1,
                segmentation_map=True, plot_wavelet=False, plot_likelihood=True,
                plot_scene=False, plot_sources=False, add_ellipses=True,
                add_labels=False,add_boxes=False):
    
    """ Run P. Melchior's scarlet (https://github.com/pmelchior/scarlet) implementation 
    for source separation. This function will create diagnostic plots, a source detection catalog, 
    and fit a model for all sources in the observation scene (image).

    Parameters
    ----------
    subtract_background : boolean
        Whether or not to estimate and subtract the background (often background is already subtracted)
        Detault is False
    plot_wavelet_transform : boolean
        Plot starlet wavelet transform and inverse transform at different scales. 
        NOTE: Not really useful at large image sizes (> ~few hundred pixels length/height)
        Default is False
    plot_detections : boolean
        Plot detection catalog results. Default is False
    plot_likelihood : boolean
        Plot likelihood as function of iterations from Blend fit function. Default is True
    plot_full_scene : boolean
        Plot full scene with the model, rendered model, observation, and residual. Default is False.
    plot_all_sources : boolean
        Plot the model, rendered model, observation, and spectrum across channels for each object. 
        WARNING: dumb to do this with a large image with many sources! Default is False
    plot_first_isolated_comp : boolean
        Plot the subtracted and isolated first (or any) starlet component. Recommended for finding a bright
        component. Default is False.


    Return
    -------
    FITS file with...
    TODO: fill this out once I get the exact fits file output generated to Colin's liking 
    """
    
    norm = scarlet.display.AsinhMapping(minimum=0, stretch=stretch, Q=Q)
        
    # Generate source catalog using wavelets
    catalog, bg_rms_hsc = make_catalog(datas, 3, wave=True, subtract_background=subtract_background)
    # If image is already background subtracted, bg_rms_hsc will be None
    if bg_rms_hsc is None:
        weights = np.ones_like(datas)
    else:
        # Set weights to all ones
        weights = np.ones_like(datas) / (bg_rms_hsc**2)[:, None, None]
    
    print("Source catalog found ", len(catalog), "objects")
    
    # Plot wavelet transform at different scales
    if plot_wavelet == True:
        _plot_wavelet(datas)
        
    # Define model frame and observations:
    model_psf = scarlet.GaussianPSF(sigma=sigma_model) #, boxsize=100)
    model_frame = scarlet.Frame(datas.shape, psf=model_psf, channels=filters)

    observation_psf = scarlet.GaussianPSF(sigma=sigma_obs)
    observation = scarlet.Observation(datas, psf=observation_psf, weights=weights, channels=filters).match(model_frame)        
          
    # Initialize starlet sources to be fit. Assume extended sources for all because 
    # we are not looking at all detections in each image
    
    # TODO: Plot chi2 vs. binned size and mag. Implement conidition if chi2 > xxx then
    # add another component until larger sources are modeled well
    print("Initializing starlet sources to be fit.")
        
    # Compute radii and spread of sources
    Rs = np.sqrt(catalog['a']**2 + catalog['b']**2)
    spread = Rs/sigma_obs
    
    # Array of chi^2 residuals computed after fit on each model
    chi2s = np.zeros(len(catalog))
    
    # Loop through detections in catalog
    starlet_sources = []
    for k, src in enumerate(catalog):

        # Is the source compact relative to the PSF?
        if spread[k] < 1:
            compact = True
        else:
            compact = False

        # Try modeling each source as a single ExtendedSource first
        new_source = scarlet.ExtendedSource(model_frame, (src['y'], src['x']), observation,
                                            K=1, thresh=morph_thresh, compact=compact)

        starlet_sources.append(new_source)
            
    # Fit scarlet blend
    starlet_blend, logL = fit_scarlet_blend(starlet_sources, observation, plot_likelihood=plot_likelihood)
    
    print("Computing residuals.")

    # Compute reduced chi^2 for each rendered sources
    for k, src in enumerate(starlet_sources):

        model = src.get_model(frame=model_frame)
        model = observation.render(model)
        res = datas - model

        # Compute in bbox only
        res = src.bbox.extract_from(res)
        chi2s[k] = np.sum(res**2)

        # Replace models with poor fits with StarletSource models
        if chi2s[k] > max_chi2:
            starlet_sources[k] = scarlet.StarletSource(model_frame, (catalog["x"][k], catalog["y"][k]), observation,
                                                       thresh=morph_thresh, starlet_thresh=0.1, full=False)

        #plt.figure(figsize=(5,5))
        #model_rgb = scarlet.display.img_to_rgb(res, norm=norm)
        #plt.title(chi2s[k])
        #plt.imshow(model_rgb)
        #plt.show()

    # If any chi2 residuals are flagged, re-fit the blend with a more complex model
    if np.any(chi2s > max_chi2):
        
        print("Re-fitting with Starlet models for poorly-fit sources.")
        
        starlet_blend, logL = fit_scarlet_blend(starlet_sources, observation, plot_likelihood=plot_likelihood)
        
        
    # Extract the deblended catalog and update the chi2 residuals
    print('Extracting deblended catalog.')
    
    catalog_deblended = []
    segmentation_masks = []
        
    for k, src in enumerate(starlet_sources):

        model = src.get_model(frame=model_frame)
        model = observation.render(model)
        res = datas - model

        # Compute in bbox only
        model = src.bbox.extract_from(model)
        res = src.bbox.extract_from(res)
        chi2s[k] = np.sum(res**2)
        
        # Run sep
        cat, _ = make_catalog(model, 1, wave=False, subtract_background=False, segmentation_map=segmentation_map)
        if segmentation_map == True:
            cat, mask = cat
            segmentation_masks.append(mask)
        catalog_deblended.append(cat)
        
    # Combine catalog named array
    #catalog_deblended = np.vstack(catalog_deblended)
    
    # Plot scene: rendered model, observations, and residuals
    if plot_scene == True:
        _plot_scene(starlet_sources, observation, norm, catalog, show_model=False, show_rendered=True,
                   show_observed=True, show_residual=True, add_labels=add_labels, add_boxes=add_boxes, add_ellipses=add_ellipses)
    
    # Plot each for each source
    if plot_sources == True:
        scarlet.display.show_sources(starlet_sources, observation, norm=norm,
                                     show_rendered=True, show_observed=True,
                                     add_boxes=add_boxes)
        plt.show()
         
    return observation, starlet_sources, model_frame, catalog, catalog_deblended, segmentation_masks