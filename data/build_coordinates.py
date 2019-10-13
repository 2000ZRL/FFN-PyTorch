# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:38:03 2019

@author: Ronglai Zuo
"""
#build coordinates


from collections import defaultdict

from absl import app
from absl import flags
from absl import logging

import h5py
import numpy as np


FLAGS=flags.FLAGS

flags.DEFINE_list('partition_volumes', ['af:../CREMI/af_C.h5:af'],
                  'Partition volumes as '
                  '<volume_name>:<volume_path>:<dataset>, where volume_path '
                  'points to a HDF5 volume, and <volume_name> is an arbitrary '
                  'label that will have to also be used during training.')
flags.DEFINE_string('coordinate_output', '../CREMI/coordinates_file_C.npy',
                    'Path to a .npz file in which to save the '
                    'coordinates.')
flags.DEFINE_list('margin', [32,42,42], '(z, y, x) tuple specifying the '
                  'number of voxels adjacent to the border of the volume to '
                  'exclude from sampling. This should normally be set to the '
                  'radius of the FFN training FoV (i.e. network FoV radius '
                  '+ deltas.')


IGNORE_PARTITION = 255


def main(argv):
  del argv  # Unused.

  totals = defaultdict(int)  # partition -> voxel count
  indices = defaultdict(list)  # partition -> [(vol_id, 1d index)]

  vol_labels = []
  vol_shapes = []
  mz, my, mx = [int(x) for x in FLAGS.margin]

  for i, partvol in enumerate(FLAGS.partition_volumes):
    name, path, dataset = partvol.split(':')
    with h5py.File(path, 'r') as f:
      partitions = f[dataset][mz:-mz, my:-my, mx:-mx]
      vol_shapes.append(partitions.shape) #[472,472,472]
      vol_labels.append(name) #validation1

      uniques, counts = np.unique(partitions, return_counts=True)
      for val, cnt in zip(uniques, counts):
        if val == IGNORE_PARTITION:
          continue

        totals[val] += cnt
        indices[val].extend(
            [(i, flat_index) for flat_index in
             np.flatnonzero(partitions == val)])

  logging.info('Partition counts:')
  for k, v in totals.items():
    logging.info(' %d: %d', k, v)

  logging.info('Resampling and shuffling coordinates.')

  max_count = max(totals.values())
  indices = np.concatenate(
      [np.resize(np.random.permutation(v), (max_count, 2)) for
       v in indices.values()], axis=0)
  np.random.shuffle(indices)

  coord_3d=[]  #(x,y,z)
  logging.info('unravelling.')
  for i, coord_idx in indices:
    z, y, x = np.unravel_index(coord_idx, vol_shapes[i])
    coord_3d.append([mx+x, my+y, mz+z])
  
    
  logging.info('unravel success')  
  coord_3d = np.resize(np.array(coord_3d), (-1,3))
  print('the number of coordinates: ', coord_3d.shape[0])

  
  np.save(FLAGS.coordinate_output, coord_3d)

  
if __name__ == '__main__':
  '''
  flags.mark_flag_as_required('margin')
  flags.mark_flag_as_required('coordinate_output')
  flags.mark_flag_as_required('partition_volumes')
  '''
  app.run(main)
  






















