# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:39:28 2019

@author: Ronglai Zuo
"""
#utils for inference

import numpy as np
from collections import deque
import weakref


def get_scored_move_offsets(deltas, prob_map, threshold=0.9):
    """Looks for potential moves for a FFN.
    
    The possible moves are determined by extracting probability map values
    corresponding to cuboid faces at +/- deltas, and considering the highest
    probability value for every face.
    
    Args:
      deltas: (z,y,x) tuple of base move offsets for the 3 axes
      prob_map: current probability map as a (z,y,x) numpy array
      threshold: minimum score required at the new FoV center for a move to be
          considered valid
            
    Yields:
      tuples of:
        score (probability at the new FoV center),
        position offset tuple (z,y,x) relative to center of prob_map
    
      The order of the returned tuples is arbitrary and should not be depended
      upon. In particular, the tuples are not necessarily sorted by score.
    """
    center = np.array(prob_map.shape) // 2
    assert center.size == 3
    # Selects a working subvolume no more than +/- delta away from the current
    # center point.
    subvol_sel = [slice(c - dx, c + dx + 1) for c, dx
                  in zip(center, deltas)]

    done = set()
    for axis, axis_delta in enumerate(deltas):
        if axis_delta == 0:
            continue
        for axis_offset in (-axis_delta, axis_delta):
            # Move exactly by the delta along the current axis, and select the face
            # of the subvolume orthogonal to the current axis.
            face_sel = subvol_sel[:]
            face_sel[axis] = axis_offset + center[axis]
            face_prob = prob_map[tuple(face_sel)]
            shape = face_prob.shape

            # Find voxel with maximum activation.
            face_pos = np.unravel_index(face_prob.argmax(), shape)
            score = face_prob[face_pos]

            # Only move if activation crosses threshold.
            if score < threshold:
                continue

            # Convert within-face position to be relative vs the center of the face.
            relative_pos = [face_pos[0] - shape[0] // 2, face_pos[1] - shape[1] // 2]
            relative_pos.insert(axis, axis_offset)
            ret = (score, tuple(relative_pos))

            if ret not in done: 
                done.add(ret)
                yield ret
        

class BaseMovementPolicy(object):
    """Base class for movement policy queues.

    The principal usage is to initialize once with the policy's parameters and
    set up a queue for candidate positions. From this queue candidates can be
    iteratively consumed and the scores should be updated in the FFN
    segmentation loop.
    """

    def __init__(self, canvas, scored_coords, deltas):
        """Initializes the policy.

        Args:
        canvas: Canvas object for FFN inference
        scored_coords: mutable container of tuples (score, zyx coord)
        deltas: step sizes as (z,y,x)
        """
        # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
        self.canvas = weakref.proxy(canvas)
        self.scored_coords = scored_coords
        self.deltas = np.array(deltas)

    def __len__(self):
        return len(self.scored_coords)
    
    def __iter__(self):
        return self
    
    def next(self):
        raise StopIteration()
    
    def append(self, item):
        self.scored_coords.append(item)
    
    def update(self, prob_map, position):
        """Updates the state after an FFN inference call.
    
        Args:
          prob_map: object probability map returned by the FFN (in logit space)
          position: postiion of the center of the FoV where inference was performed
          (z, y, x)
        """
        raise NotImplementedError()
    
    def get_state(self):
        """Returns the state of this policy as a pickable Python object."""
        raise NotImplementedError()
    
    def restore_state(self, state):
        raise NotImplementedError()
    
    def reset_state(self, start_pos):
        """Resets the policy.
    
        Args:
        start_pos: starting position of the current object as z, y, x
        """
        raise NotImplementedError()


class FaceMaxMovementPolicy(BaseMovementPolicy):
    """Selects candidates from maxima on prediction cuboid faces."""

    def __init__(self, canvas, deltas=(4, 8, 8), score_threshold=0.9):
        self.done_rounded_coords = set()  #save visited coords
        self.score_threshold = score_threshold
        self._start_pos = None
        super(FaceMaxMovementPolicy, self).__init__(canvas, deque([]), deltas)

    def reset_state(self, start_pos):
        self.scored_coords = deque([])
        self.done_rounded_coords = set()
        self._start_pos = start_pos

    def get_state(self):
        return [(self.scored_coords, self.done_rounded_coords)]

    def restore_state(self, state):
        self.scored_coords, self.done_rounded_coords = state[0]

    def __next__(self):
        """Pops positions from queue until a valid one is found and returns it."""
        while self.scored_coords:
             _, coord = self.scored_coords.popleft()
             coord = tuple(coord)
             if self.quantize_pos(coord) in self.done_rounded_coords:
                 continue
             if self.canvas.is_valid_pos(coord):
                 break
        else:  # Else goes with while, not with if!
            raise StopIteration()

        return tuple(coord)

    def next(self):
        return self.__next__()

    def quantize_pos(self, pos):  
        """Quantizes the positions symmetrically to a grid downsampled by deltas."""
        # Compute offset relative to the origin of the current segment and
        # shift by half delta size. This ensures that all directions are treated
        # approximately symmetrically -- i.e. the origin point lies in the middle of
        # a cell of the quantized lattice, as opposed to a corner of that cell.
        rel_pos = (np.array(pos) - self._start_pos)
        coord = (rel_pos + self.deltas // 2) // np.maximum(self.deltas, 1)
        return tuple(coord)

    def update(self, prob_map, position):
        """Adds movements to queue for the cuboid face maxima of ``prob_map``."""
        qpos = self.quantize_pos(position)
        self.done_rounded_coords.add(qpos)
    
        scored_coords = get_scored_move_offsets(self.deltas, prob_map,
                                                threshold=self.score_threshold)
        scored_coords = sorted(scored_coords, reverse=True)
        for score, rel_coord in scored_coords:
            # convert to whole cube coordinates
            coord = [rel_coord[i] + position[i] for i in range(3)]
            self.scored_coords.append((score, coord))
