# -*- coding: utf-8 -*-
import numpy as np


def quantize_probability(prob):
    """Quantizes a probability map into a byte array."""
    ret = np.digitize(prob, np.linspace(0.0, 1.0, 255))

    # Digitize never uses the 0-th bucket.
    ret[np.isnan(prob)] = 0
    return ret.astype(np.uint8)

def reduce_id_bits(segmentation):
    """Reduces the number of bits used for IDs.

    Assumes that one additional ID beyond the max of 'segmentation' is necessary
    (used by GALA to mark boundary areas).

    Args:
      segmentation: ndarray of int type

    Returns:
      segmentation ndarray converted to minimal uint type large enough to keep
      all the IDs.
    """
    max_id = segmentation.max()
    if max_id <= np.iinfo(np.uint8).max:
        return segmentation.astype(np.uint8)
    elif max_id <= np.iinfo(np.uint16).max:
        return segmentation.astype(np.uint16)
    elif max_id <= np.iinfo(np.uint32).max:
        return segmentation.astype(np.uint32)