import cv2
import numpy as np
from astropy.io import fits
from detectron2.structures import BoxMode


def annotate_decam(images, mask, idx):
    """Generates annotation metadata for decam data

    Parameters
    ----------
    images : list
        A list of paths to image files, expected to have one file per filter.
    mask: str
        A path to a mask file for the images.
    idx: int
        An integer to uniquely identify the resulting record.

    Returns
    -------
    record : dictionary
        A dictionary of metadata and derived annotations.
    """
    record = {}

    # Open each FITS image
    with fits.open(mask, memmap=False, lazy_load_hdus=False) as hdul:
        sources = len(hdul)
        height, width = hdul[0].data.shape
        data = [hdu.data / np.max(hdu.data) for hdu in hdul]
        category_ids = [hdu.header["CLASS_ID"] for hdu in hdul]

    record["file_name"] = images[0][0:-7]  # chop off the _[band].fits
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width
    objs = []

    # Mask value thresholds per category_id
    thresh = [0.005 if i == 1 else 0.08 for i in category_ids]

    # Generate segmentation masks
    for i in range(sources):
        image = data[i]
        mask = np.zeros([height, width], dtype=np.uint8)
        # Create mask from threshold
        mask[:, :][image > thresh[i]] = 1
        # Smooth mask
        mask[:, :] = cv2.GaussianBlur(mask[:, :], (9, 9), 2)

        # https://github.com/facebookresearch/Detectron/issues/100
        contours, _ = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour = contour.flatten().tolist()
            # segmentation.append(contour)
            if len(contour) > 4:
                segmentation.append(contour)
        # No valid contours
        if len(segmentation) == 0:
            continue

        # Add to dict
        obj = {
            "bbox": [x, y, w, h],
            "area": w * h,
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": segmentation,
            "category_id": category_ids[i] - 1,
        }
        objs.append(obj)

    record["annotations"] = objs
    return record
