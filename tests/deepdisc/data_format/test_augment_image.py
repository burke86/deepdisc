"""Utilities for augmenting image data."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from deepdisc.data_format.augment_image import (addelementwise,
                                                addelementwise8,
                                                addelementwise16, centercrop,
                                                gaussblur)


@pytest.fixture
def simple_image():
    input_image = np.arange(0, 100, dtype=np.int16)
    input_image = input_image.reshape(10, 10)
    return input_image


def test_gaussblur(simple_image):
    output = gaussblur(simple_image, rng_seed=np.random.default_rng(54622))
    assert len(output) == len(simple_image)
    expected = [
        [10, 11, 11, 12, 13, 14, 15, 16, 17, 17],
        [13, 14, 14, 15, 16, 17, 18, 19, 20, 20],
        [21, 21, 22, 23, 24, 25, 26, 27, 28, 28],
        [31, 31, 32, 33, 34, 35, 36, 37, 38, 38],
        [41, 41, 42, 43, 44, 45, 46, 47, 48, 48],
        [51, 51, 52, 53, 54, 55, 56, 57, 58, 58],
        [61, 61, 62, 63, 64, 65, 66, 67, 68, 68],
        [71, 71, 72, 73, 74, 75, 76, 77, 78, 78],
        [79, 79, 80, 81, 82, 83, 84, 85, 85, 86],
        [82, 82, 83, 84, 85, 86, 87, 88, 88, 89],
    ]
    assert np.all(output == expected)


def test_addelementwise16(simple_image):
    output = addelementwise16(simple_image, rng_seed=np.random.default_rng(54622))
    expected_first = [260, -1743, 1504, -2360, 409, -1053, 163, 266, 2811, -337]
    assert np.all(output[0] == expected_first)
    assert len(output) == len(simple_image)


def test_addelementwise8(simple_image):
    output = addelementwise8(simple_image, rng_seed=np.random.default_rng(54622))
    expected_first = [2, -12, 13, -15, 7, -3, 7, 9, 29, 6]
    assert np.all(output[0] == expected_first)
    assert len(output) == len(simple_image)


def test_addelementwise(simple_image):
    output = addelementwise(simple_image, rng_seed=np.random.default_rng(54622))
    expected_first = [1, -4, 7, -4, 5, 2, 6, 8, 16, 8]
    assert np.all(output[0] == expected_first)
    assert len(output) == len(simple_image)


def test_centercrop(simple_image):
    output = centercrop(simple_image)
    assert len(output) == len(simple_image) / 2
    expected = [
        [22, 23, 24, 25, 26],
        [32, 33, 34, 35, 36],
        [42, 43, 44, 45, 46],
        [52, 53, 54, 55, 56],
        [62, 63, 64, 65, 66],
    ]
    assert np.all(output == expected)
