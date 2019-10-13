# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:55:56 2019

@author: Ronglai Zuo
"""

#data preprocess, the algorithm is the same as both compute_partitions.py & build_coordinates.py


import torch as t
from torch.autograd import Variable as V
import numpy as np
import h5py

from absl import app
from absl import flags
from absl import logging

"""
fov_size = np.array([33,33,33],dtype=np.int)
delta=np.array([8,8,8],dtype=np.int)
lom_radius=fov_size//2 + delta
"""

FLAGS=flags.FLAGS

flags.DEFINE_string('input_volume', '/mnt/dive/shared/yaochen.xie/EMImage/CREMI/train/cropped/sample_C_20160501.hdf:volumes/labels/neuron_ids',
                    'Segmentation volume as <volume_path>:<dataset>, where'
                    'volume_path points to a HDF5 volume.')
flags.DEFINE_string('output_volume', '../CREMI/af_C.h5:af',
                    'Volume in which to save the partition map, as '
                    '<volume_path>:<dataset>.')
flags.DEFINE_list('thresholds', [0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                  'List of activation voxel fractions used for partitioning.')
flags.DEFINE_list('lom_radius', [16,16,16],
                  'Local Object Mask (LOM) radii as (x, y, z).')
flags.DEFINE_integer('min_size', 10000,
                     'Minimum number of voxels for a segment to be considered for '
                     'partitioning.')
"""
flags.DEFINE_list('id_whitelist', None,
                  'Whitelist of object IDs for which to compute the partition '
                  'numbers.')
flags.DEFINE_list('exclusion_regions', None,
                  'List of (x, y, z, r) tuples specifying spherical regions to '
                  'mark as excluded (i.e. set the output value to 255).')
flags.DEFINE_string('mask_configs', None,
                    'MaskConfigs proto in text foramt. Any locations where at '
                    'least one voxel of the LOM is masked will be marked as '
                    'excluded.')
"""


def clear_dust(data, min_size):
    #clear little objects
    ids, sizes = np.unique(data, return_counts=True)
    small = ids[sizes < min_size]
    small_mask = np.in1d(data.flat, small).reshape(data.shape)
    data[small_mask] = 0
    return data


def _summed_volume_table(val):
    """Computes a summed volume table of 'val'."""
    val = val.astype(np.int32)
    svt = val.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    return np.pad(svt, [[1, 0], [1, 0], [1, 0]], mode='constant')


def _query_summed_volume(svt, diam):
    """Queries a summed volume table.

    Operates in 'VALID' mode, i.e. only computes the sums for voxels where the
    full diam // 2 context is available.

    Args:
      svt: summed volume table (see _summed_volume_table)
      diam: diameter (z, y, x tuple) of the area within which to compute sums

    Returns:
      sum of all values within a diam // 2 radius (under L1 metric) of every voxel
      in the array from which 'svt' was built.
    """
    return (
        svt[diam[0]:, diam[1]:, diam[2]:] - svt[diam[0]:, diam[1]:, :-diam[2]] -
        svt[diam[0]:, :-diam[1], diam[2]:] - svt[:-diam[0], diam[1]:, diam[2]:] +
        svt[:-diam[0], :-diam[1], diam[2]:] + svt[:-diam[0], diam[1]:, :-diam[2]]
        + svt[diam[0]:, :-diam[1], :-diam[2]] -
        svt[:-diam[0], :-diam[1], :-diam[2]])
    
    
def compute_partitions(seg_array, thresholds, lom_radius, min_size=10000):
    
    seg_array = clear_dust(seg_array, min_size=min_size)
    assert seg_array.ndim == 3

    lom_radius = np.array(lom_radius)
    lom_radius_zyx = lom_radius[::-1]
    lom_diam_zyx = 2 * lom_radius_zyx + 1

    def _sel(i):
        if i == 0:
            return slice(None)
        else:
            return slice(i, -i)

    valid_sel = [_sel(x) for x in lom_radius_zyx]
    output = np.zeros(seg_array[valid_sel].shape, dtype=np.uint8)
    corner = lom_radius

    labels = set(np.unique(seg_array))
    logging.info('Labels to process: %d', len(labels))

    fov_volume = np.prod(lom_diam_zyx)
    counter=0
    for l in labels:
        # Don't create a mask for the background component.
        if l == 0:
            continue

        object_mask = (seg_array == l)

        svt = _summed_volume_table(object_mask)
        active_fraction = _query_summed_volume(svt, lom_diam_zyx) / fov_volume
        assert active_fraction.shape == output.shape

        # Drop context that is only necessary for computing the active fraction
        # (i.e. one LOM radius in every direction).
        object_mask = object_mask[valid_sel]  #扣去了lom_radius=24的elements

        # TODO(mjanusz): Use np.digitize here.
        for i, th in enumerate(thresholds):
            output[object_mask & (active_fraction < th) & (output == 0)] = i + 1

        output[object_mask & (active_fraction >= thresholds[-1]) & (output == 0)] = len(thresholds) + 1
        counter += 1
        logging.info('Done processing %d  processed: %f', l, counter/len(labels))

    logging.info('Nonzero values: %d', np.sum(output > 0))
    print(output.shape)

    return corner, output    


def main(argv):
    del argv  # Unused.
    path, dataset = FLAGS.input_volume.split(':')
    with h5py.File(path) as f:
        segmentation = f[dataset]
        """
      bboxes = []
      for name, v in segmentation.attrs.items():
          if name.startswith('bounding_boxes'):
              for bbox in v:
                  bboxes.append(bounding_box.BoundingBox(bbox[0], bbox[1]))

      if not bboxes:
          bboxes.append(
              bounding_box.BoundingBox(
                  start=(0, 0, 0), size=segmentation.shape[::-1]))
        """
        shape = segmentation.shape
        lom_radius = [int(x) for x in FLAGS.lom_radius]
        corner, partitions = compute_partitions(
            segmentation[...], [float(x) for x in FLAGS.thresholds], lom_radius,
            FLAGS.min_size)

    #bboxes = adjust_bboxes(bboxes, np.array(lom_radius))

    path, dataset = FLAGS.output_volume.split(':')
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset(dataset, shape=shape, dtype=np.uint8, fillvalue=255,
                              chunks=True, compression='gzip')
        s = partitions.shape
        ds[corner[2]:corner[2] + s[0],
            corner[1]:corner[1] + s[1],
            corner[0]:corner[0] + s[2]] = partitions  #边缘处灰度值不变
        """
        ds.attrs['bounding_boxes'] = [(b.start, b.size) for b in bboxes]
        ds.attrs['partition_counts'] = np.array(np.unique(partitions,
                                                          return_counts=True))
        """


if __name__ == '__main__':
    '''
    flags.mark_flag_as_required('input_volume')
    flags.mark_flag_as_required('output_volume')
    flags.mark_flag_as_required('thresholds')
    flags.mark_flag_as_required('lom_radius')
    '''
    app.run(main)



















