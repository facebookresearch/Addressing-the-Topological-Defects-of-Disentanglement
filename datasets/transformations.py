"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Transformations applied to the input images
"""

import torch
import itertools
import numpy as np
import skimage.transform
from dataclasses import dataclass


# TODO: set automaticlaly based on n_pixels
TRANSLATION_INTERVAL = [0, 28]


@dataclass
class Params:
    """
        angle (float): counter-clockwise rotation angle in degrees
        shift_x (float): shift value to the right
        shift_y (float): shift value to upwards
        scale (float): scaling factor
    """

    angle: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    scale: float = 1.0


def transform(image, params):
    """
    Applies transformations on a single image based on params.
    Order of transformation is: rotate, translate, scale

    Args:
        image (np.array or torch.tensor): of shape [n_pixels, n_pixels]
        params (Params): contains parameters for rotations, scaling etc.

    Returns: image with transformations applied
    """
    assert (
        image.ndim == 3
    ), f"image must be of shape [n_channels, n_pixels, n_pixels] not {image.shape}"

    image_transformed = image.squeeze()
    # Rotate
    if params.angle not in (0.0, 360.0):
        # cval is the fill value.
        image_transformed = skimage.transform.rotate(
            image_transformed, params.angle, cval=image_transformed.min()
        )

    # Translate
    # if edge is reached cut-off portion appears on other side
    if params.shift_x != 0.0:
        image_transformed = np.roll(image_transformed, int(params.shift_x), axis=1)
    if params.shift_y != 0.0:
        image_transformed = np.roll(image_transformed, -int(params.shift_y), axis=0)

    # Scale
    if params.scale != 1.0:
        image_transformed = rescale(image_transformed, params.scale)
    image_transformed = to_torch(image, image_transformed)
    return image_transformed


def rescale(image, scale):
    """Rescales images based on given scale factor"""
    scale_transform = skimage.transform.SimilarityTransform(scale=scale)
    image = skimage.transform.warp(
        image, scale_transform.inverse, mode="constant", cval=image.min(),
    )
    return image


def to_torch(image, image_transformed):
    """Converts numpy matrix to torch tensor with correct shape"""
    image_transformed = image_transformed.reshape(image.shape)
    if torch.is_tensor(image_transformed):
        return image_transformed.float()
    if torch.is_tensor(image):
        image_transformed = torch.from_numpy(image_transformed).float()
    return image_transformed


def get_transform_params(
    n_rotations, n_x_translations, n_y_translations, scaling_factors,
):
    """Returns transform params corresponding given values.
    Translations subdivide translation interval.

    Args:
        n_rotations (int): number of subdivisions of 360 to apply. 
        n_x_translations (int): number of shifts along x-axis
        n_y_translations (int): number of shifts along y-axis
        scaling_factors (list or tuple floats): representing the scaling factors to use

    Returns: Params object
    """
    shifts_x = get_shifts(n_x_translations, TRANSLATION_INTERVAL)
    shifts_y = get_shifts(n_y_translations, TRANSLATION_INTERVAL)

    for angle in get_rotation_angles(n_rotations):
        for shift_x, shift_y in itertools.product(shifts_x, shifts_y):
            for scale in scaling_factors:
                params = Params(
                    angle=angle, shift_x=shift_x, shift_y=shift_y, scale=scale
                )
                yield params


def get_shifts(n_translations, interval):
    """Returns shifts along given axis by dividing interval.

    Args:
        interval (list of ints): [0, n_pixels]
        n_translations (int): should be divisible by n_pixels
    """
    if n_translations == 0:
        return [0]
    elif n_translations == 1:
        return [0, interval[1] // 2]

    min_shift = round(interval[1] / (n_translations + 1))
    steps = [n * min_shift for n in range(n_translations + 1)]

    return steps


def get_rotation_angles(n_rotations):
    """Yields rotation angles based on subdivisions given.
    Example: 
    >>> get_rotation_angles(2) => [0.0, 120.0, 240.0]
    """
    min_angle = 360.0 / (n_rotations + 1)
    for n in range(n_rotations + 1):
        yield min_angle * n


def shift_to_angle(shift_val, n_transformations):
    """Returns the angle corresponding to the shift_val.
    Example: [0, 32], shift_val = 4, we should get 4 / 32 * 360
    """
    if shift_val == TRANSLATION_INTERVAL[1]:
        return 0.0
    shift_ratio = float(shift_val) / TRANSLATION_INTERVAL[1]
    angle = 360.0 * shift_ratio
    return angle
