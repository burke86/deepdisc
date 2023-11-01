"""Utilities for augmenting image data."""

import detectron2.data.transforms as T
import imgaug.augmenters as iaa
import numpy as np

import deepdisc.astrodet.detectron as detectron_addons


def gaussblur(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    aug = iaa.GaussianBlur(sigma=(0.0, rng_seed.random() * 4 + 2), seed=rng_seed)
    return aug.augment_image(image)


def addelementwise16(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    aug = iaa.AddElementwise((-3276, 3276), seed=rng_seed)
    return aug.augment_image(image)


def addelementwise8(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    aug = iaa.AddElementwise((-25, 25), seed=rng_seed)
    return aug.augment_image(image)


def addelementwise(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    aug = iaa.AddElementwise((-image.max() * 0.1, image.max() * 0.1), seed=rng_seed)
    return aug.augment_image(image)


def centercrop(image):
    """Crop an image to just the center portion

    Parameters
    ----------
    image: ndarray

    Returns
    -------
    cropped image
    """
    h, w = image.shape[:2]
    hc = (h - h // 2) // 2
    wc = (w - w // 2) // 2
    image = image[hc : hc + h // 2, wc : wc + w // 2]
    return image


def train_augs(image):
    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ],
        k=-1,
        cropaug=T.RandomCrop("relative", (0.5, 0.5)),
    )
    return augs


def hsc_test_augs(image):
    augs = T.AugmentationList(
        [
            T.CropTransform(
                image.shape[1] // 4,
                image.shape[0] // 4,
                image.shape[1] // 2,
                image.shape[0] // 2,
            )
        ]
    )
    return augs
