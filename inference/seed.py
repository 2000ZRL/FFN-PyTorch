# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:01:42 2019

@author: Ronglai Zuo
"""

import logging
import weakref
from collections import OrderedDict
import numpy as np
from scipy import ndimage
import skimage
import skimage.feature


class BaseSeedPolicy(object):
    """Base class for seed policies."""

    def __init__(self, canvas, **kwargs):
        """Initializes the policy.

        Args:
          canvas: inference Canvas object; simple policies use this to access
              basic geometry information such as the shape of the subvolume;
              more complex policies can access the raw image data, etc.
          **kwargs: other keyword arguments
        """
        del kwargs
        # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
        self.canvas = weakref.proxy(canvas)
        self.coords = None
        self.idx = 0

    def _init_coords(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next seed point as (z, y, x).

        Does initial filtering of seed points to exclude locations that are
        too close to the image border.
    
        Returns:
          (z, y, x) tuples.
    
        Raises:
          StopIteration when the seeds are exhausted.
        """
        if self.coords is None:
            self._init_coords()

        while self.idx < self.coords.shape[0]:
            curr = self.coords[self.idx, :]
            self.idx += 1

            # TODO(mjanusz): Get rid of this.
            # Do early filtering of clearly invalid locations (too close to image
            # borders) as late filtering might be expensive.
            if (np.all(curr - self.canvas.margin >= 0) and
                np.all(curr + self.canvas.margin < self.canvas.shape)):
                return tuple(curr)  # z, y, x
            
        raise StopIteration()

    def next(self):
        return self.__next__()

    def get_state(self):
        return self.coords, self.idx

    def set_state(self, state):
        self.coords, self.idx = state


class PolicyPeaks(BaseSeedPolicy):
    """Attempts to find points away from edges in the image.

    Runs a 3d Sobel filter to detect edges in the raw data, followed
    by a distance transform and peak finding to identify seed points.
    """

    def _init_coords(self):
        logging.info('peaks: starting')
  
        # Edge detection.
        edges = ndimage.generic_gradient_magnitude(
            self.canvas.image.astype(np.float32),
            ndimage.sobel)

        # Adaptive thresholding.
        sigma = 49.0 / 6.0
        thresh_image = np.zeros(edges.shape, dtype=np.float32)
        ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect') #edge --gauss--> thresh
        filt_edges = edges > thresh_image
    
        del edges, thresh_image
        '''
        # This prevents a border effect where the large amount of masked area
        # screws up the distance transform below.
        if (self.canvas.restrictor is not None and
            self.canvas.restrictor.mask is not None):
          filt_edges[self.canvas.restrictor.mask] = 1
        '''
        logging.info('peaks: filtering done')
        dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32) #离边缘的距离
        logging.info('peaks: edt done')

        # Use a specifc seed for the noise so that results are reproducible
        # regardless of what happens before the policy is called.
        state = np.random.get_state()
        np.random.seed(42)
        idxs = skimage.feature.peak_local_max(
            dt + np.random.random(dt.shape) * 1e-4,
            indices=True, min_distance=3, threshold_abs=0, threshold_rel=0) #离边缘距离的peak
        np.random.set_state(state)
        
        logging.info('sorting...')
        idxs_dict = OrderedDict()
        for z,y,x in idxs:
            idxs_dict[str(z)+'-'+str(y)+'-'+str(x)] = dt[z][y][x]
        idxs_dict = OrderedDict(sorted(idxs_dict.items(), key=lambda x: x[1], reverse=True))

        # After skimage upgrade to 0.13.0 peak_local_max returns peaks in
        # descending order, versus ascending order previously.  Sort ascending to
        # maintain historic behavior.
        idxs = []
        for key in idxs_dict.keys():
            z,y,x=key.split('-')
            idxs.append((int(z),int(y),int(x)))
        idxs = np.reshape(idxs, (-1,3))
        logging.info('peaks: found %d local maxima', idxs.shape[0])
        self.coords = idxs
