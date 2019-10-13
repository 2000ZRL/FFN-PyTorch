# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:39:26 2019

@author: Ronglai Zuo
"""
#infernce for FFN

import numpy as np
import torch as t
import logging
from models import basic_model_11 as basic_model
from scipy.special import logit
from scipy.special import expit
from inference import movement
from inference import seed
from inference import utils

from absl import flags
from absl import app
from time import time
import h5py
import os


FLAGS = flags.FLAGS

flags.DEFINE_integer('image_mean', 128, 'image_mean')
flags.DEFINE_integer('image_stddev', 33, 'image_stddev')
flags.DEFINE_string('image_path', 'FIB-25/test_sample/grayscale_maps.h5:raw', 'image_path')
flags.DEFINE_string('checkpoints_path', 'results/checkpoints/seg_ckp', 'checkpoints_path')
flags.DEFINE_string('seg_result_path', 'results/test_sample/basic11/', 'seg_result_path')

flags.DEFINE_string('base_dir', 'checkpoints/originalFFN/basic11/', 'model_base_dir')
flags.DEFINE_string('model_name', None, 'model_name')

flags.DEFINE_string('gpu_id', '0', 'gpuid')
flags.DEFINE_float('init_act', 0.95, 'init_activation')
flags.DEFINE_float('pad_value', 0.05, 'pad_value')
flags.DEFINE_float('move_threshold', 0.9, 'move_threshold')
flags.DEFINE_list('min_bound_dist', [1,1,1], 'min_boundary_dist_xyz')
flags.DEFINE_float('seg_threshold', 0.6, 'segment_threshold')
flags.DEFINE_float('diff_threshold', 0.3, 'diff_threshold')
flags.DEFINE_integer('min_seg_size', 1000, 'min_segment_size')
flags.DEFINE_list('fov_size', [33,33,33], 'fov')
flags.DEFINE_list('deltas', [8,8,8], 'deltas')
flags.DEFINE_integer('ckp_interval', 1200, 'checkpoint_interval')

flags.DEFINE_integer('MAX_SELF_CONSISTENT_ITERS', 32, '...')

flags.DEFINE_list('corner', [0,0,0], 'corner')
flags.DEFINE_list('subvol_size', [520,520,520], 'subvol_size')



class Canvas(object):
    
    def __init__(self, model, image, corner, movement_policy=None):
        self.model = model.cuda()
        self.image = image
        self.corner = corner  #zyx
        self.options = {'ckp_path': FLAGS.checkpoints_path,
                        'ckp_interval': FLAGS.ckp_interval,
                        'init_act': logit(FLAGS.init_act),
                        'pad_value': logit(FLAGS.pad_value),
                        'move_threshold': logit(FLAGS.move_threshold),
                        'seg_threshold': logit(FLAGS.seg_threshold),
                        'min_seg_size': FLAGS.min_seg_size,
                        'fov_size': FLAGS.fov_size,
                        'deltas': FLAGS.deltas,
                        'min_bound_dist': FLAGS.min_bound_dist,
                        'diff_threshold': FLAGS.diff_threshold}
        
        self.shape = image.shape
        self._pred_size = self._input_image_size = self._input_seed_size = np.array(self.options['fov_size'])
        self.margin = self._input_image_size // 2  #zyx
        self.seed = np.zeros(self.shape, dtype=np.float32)
        self.segmentation = np.zeros(self.shape, dtype=np.int32)
        self.seg_prob = np.zeros(self.shape, dtype=np.uint8)

        self._max_id = 0
        # Whether to always create a new seed in segment_at.
        self.reset_seed_per_segment = True
        self.overlaps = {}
        self.origins = {}
        self.checkpoint_last = time()
        
        if movement_policy is None:
            self.movement_policy = movement.FaceMaxMovementPolicy(self, 
                                                                  deltas=tuple(self.options['deltas']),
                                                                  score_threshold=self.options['move_threshold'])
        self.reset((0,0,0))
    
    def reset(self, start_pos):
        self._min_pos = np.array(start_pos)
        self._max_pos = np.array(start_pos)
        self.movement_policy.reset_state(start_pos)
        
    def init_seed(self, pos):
        self.seed[...] = np.nan
        self.seed[pos] = self.options['init_act']
        
    def is_valid_pos(self, pos, ignore_move_threshold=False):
        """Returns True if segmentation should be attempted at the given position.

        Args:
          pos: position to check as (z, y, x)
          ignore_move_threshold: (boolean) when starting a new segment at pos the
              move threshold can and must be ignored.
    
        Returns:
          Boolean indicating whether to run FFN inference at the given position.
        """

        if not ignore_move_threshold:
           if self.seed[pos] < self.options['move_threshold']:
               logging.debug('.. seed value below threshold.')
               return False

        # Not enough image context?
        np_pos = np.array(pos)
        low = np_pos - self.margin  #marigin=16,16,16
        high = np_pos + self.margin

        if np.any(low < 0) or np.any(high >= self.shape):  #越界
            logging.debug('.. too close to border: %r', pos)
            return False

        # Location already segmented?
        if self.segmentation[pos] > 0:  #已分割
            logging.debug('.. segmentation already active: %r', pos)
            return False

        return True
    
    def predict(self, pos, logit_seed):
        '''Run a single step of FFN, and the centre is pos '''
#        from skimage.measure import label   
#        def getLargestCC(segmentation):
#            labels = label(segmentation)
#            assert( labels.max() != 0 ) 
#            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
#            return largestCC
        
        start = np.array(pos) - self.margin
        end = start + self._input_image_size
        sel = [slice(s, e) for s, e in zip(start, end)]
        img = self.image[tuple(sel)]
        img = np.reshape(img, [1,1] + self._input_image_size.tolist())
        logit_seed = np.reshape(logit_seed, [1,1] + self._input_seed_size.tolist())
        
        model_input = t.from_numpy(np.concatenate((img, logit_seed), axis=1)).cuda()
        output = self.model(model_input)
        output = output.detach().cpu().numpy()
        logits = output + logit_seed
        prob = expit(logits)
        '''
        try:
            mask = getLargestCC(logits > self.options['seg_threshold'])
            logits = mask*logits
            prob = mask*prob
        except:
            pass
        '''
        return (logits[0, 0, ...], prob[0, 0, ...])
    
    def update_at(self, pos):
        off = self._input_seed_size // 2
        
        start = np.array(pos) - off
        end = start + self._input_seed_size
        sel = [slice(s, e) for s, e in zip(start, end)]
        logit_seed = np.array(self.seed[tuple(sel)])
        init_prediction = np.isnan(logit_seed)
        logit_seed[init_prediction] = np.float32(self.options['pad_value']) #nan->pad_value
        prob_seed = expit(logit_seed)
        #(logits, prob) = self.predict(pos, logit_seed)
        
        if self.options['diff_threshold'] > 0:
            for i in range(FLAGS.MAX_SELF_CONSISTENT_ITERS):
                (logits, prob) = self.predict(pos, logit_seed)

                diff = np.average(np.abs(prob_seed - prob))
                if diff < self.options['diff_threshold']:
                    break

                prob_seed, logit_seed = prob, logits
        else:
            (logits, prob) = self.predict(pos, logit_seed)
            
        sel = [slice(s, e) for s, e in zip(start, end)]
        sel = tuple(sel)
        
        # Bias towards oversegmentation by making it impossible to reverse
        # disconnectedness predictions in the course of inference.
        th_max = logit(0.5)
        old_seed = self.seed[sel]
        if (np.mean(logits >= self.options['move_threshold']) > 0):
            # Because (x > NaN) is always False, this mask excludes positions that
            # were previously uninitialized (i.e. set to NaN in old_seed).
            try:
                old_err = np.seterr(invalid='ignore')
                mask = ((old_seed < th_max) & (logits > old_seed))
            finally:
                np.seterr(**old_err)
            logits[mask] = old_seed[mask]
        
        #updating...
        self.seed[sel] = logits
        
        return logits
    
    def segment_at(self, start_pos):
        if self.reset_seed_per_segment:
            self.init_seed(start_pos)
        self.reset(start_pos)
        num_iters = 0
        #initialize
        item = (self.movement_policy.score_threshold * 2, start_pos)
        self.movement_policy.append(item)
        
        for pos in self.movement_policy:
            if self.seed[start_pos] < self.options['move_threshold']:
                logging.info('seed got too weak..')
                break
            
            pred = self.update_at(pos)
            self._min_pos = np.minimum(self._min_pos, pos)
            self._max_pos = np.maximum(self._max_pos, pos)
            num_iters += 1

            self.movement_policy.update(pred, pos)
            
            #save checkpoints...
            if time() - self.checkpoint_last >= self.options['ckp_interval']:
                self.save_checkpoint(self.options['ckp_path']+str(start_pos)+str(num_iters))
                self.checkpoint_last = time()
            
        return num_iters
    
    def segment_all(self, seed_policy=seed.PolicyPeaks):
        self.seed_policy = seed_policy(self)
        mbd = np.array(self.options['min_bound_dist'][::-1])
        
        for pos in self.seed_policy:  #actually, start_pos
            if not self.is_valid_pos(pos, ignore_move_threshold=True):
                continue
            
            #save checkpoints...
            if time() - self.checkpoint_last >= self.options['ckp_interval']:
                self.save_checkpoint(self.options['ckp_path']+str(pos)+'0')
                self.checkpoint_last = time()
                
            # Too close to an existing segment?
            low = np.array(pos) - mbd
            high = np.array(pos) + mbd + 1
            sel = [slice(s, e) for s, e in zip(low, high)]
            sel = tuple(sel)
            if np.any(self.segmentation[sel] > 0):  #1，1，1范围内已分割，invalid -> continue
                logging.debug('Too close to existing segment.')
                self.segmentation[pos] = -1
                continue

            logging.info('Starting segmentation at %r (zyx)', pos)
            # Try segmentation.
            seg_start = time()
            num_iters = self.segment_at(pos)
            t_seg = time() - seg_start
            logging.info('time cost of one segmentation: %f', t_seg)
            logging.info('time cost of one iter: %f', t_seg/num_iters)
            # Check if segmentation was successful.
            if num_iters <= 0:
                logging.info('Failed: num iters was %d', num_iters)
                continue
            
            # Original seed too weak?
            if self.seed[pos] < self.options['move_threshold']:
                # Mark this location as excluded.
                if self.segmentation[pos] == 0:
                    self.segmentation[pos] = -1
                logging.info('Failed: weak seed')
                continue
            
            # Restrict probability map -> segment processing to a bounding box
            # covering the area that was actually changed by the FFN. In case the
            # segment is going to be rejected due to small size, this can
            # significantly reduce processing time.
            sel = [slice(max(s, 0), e + 1) for s, e in zip(
                self._min_pos - self._pred_size // 2,
                self._max_pos + self._pred_size // 2)]
            sel = tuple(sel)
            # We only allow creation of new segments in areas that are currently
            # empty.
            mask = self.seed[sel] >= self.options['seg_threshold']
            #raw_segmented_voxels = np.sum(mask)
            
            # Record existing segment IDs overlapped by the newly added object.
            overlapped_ids, counts = np.unique(self.segmentation[sel][mask],
                                               return_counts=True)
            valid = overlapped_ids > 0
            overlapped_ids = overlapped_ids[valid]
            counts = counts[valid]

            mask &= self.segmentation[sel] <= 0
            actual_segmented_voxels = np.sum(mask)
            
            if actual_segmented_voxels < self.options['min_seg_size']:
                if self.segmentation[pos] == 0:
                    self.segmentation[pos] = -1
                logging.info('Failed: too small: %d', actual_segmented_voxels)
                continue
            
            # Find the next free ID to assign.
            self._max_id += 1
            while self._max_id in self.origins:
                self._max_id += 1
                
            self.segmentation[sel][mask] = self._max_id
            self.seg_prob[sel][mask] = utils.quantize_probability(
                expit(self.seed[sel][mask]))
            
            logging.info('Created supervoxel:%d  seed(zyx):%s  size:%d  iters:%d',
                         self._max_id, pos,
                         actual_segmented_voxels, num_iters)
            
            self.overlaps[self._max_id] = np.array([overlapped_ids, counts])
            self.origins[self._max_id] = str(pos) + str(num_iters) + str(t_seg)
            
        logging.info('segmentation done')
        
    def save_checkpoint(self, path):
        """Saves a inference checkpoint to `path`."""
        logging.info('Saving inference checkpoint to %s', self.options['ckp_path'])
        np.savez_compressed(path,
                            segmentation = self.segmentation,
                            seg_quanprob = self.seg_prob,
                            seed = self.seed,
                            origins = self.origins,
                            overlaps = self.overlaps)
        logging.info('Inference checkpoint saved')
        
        
def load_network(network, model_path):
    save_path = model_path
    state_dict = t.load(save_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] # remove `module.`
        new_state_dict[namekey] = v
    # load params
    network.load_state_dict(new_state_dict)
    return network

def make_canvas(corner, subvol_size, model, image_path):
    path, dataset = image_path.split(':')
    logging.info('reading images...')
    with h5py.File(path, 'r') as raw_images:
        whole_images = raw_images[dataset][:]
    sel = [slice(s, e) for s, e in zip(corner, subvol_size)]
    img = np.array(whole_images[tuple(sel)], dtype=np.float32)
    
    logging.info('image shape: %r', img.shape)
    img = (img - FLAGS.image_mean) / FLAGS.image_stddev
    return Canvas(model, img, corner)

def save_segmentation(corner, canvas, seg_result_path):
    # Remove markers.
    canvas.segmentation[canvas.segmentation < 0] = 0
    
    # Save segmentation results. Reduce # of bits per item if possible.
    logging.info('saving segmentation results...')
    seg = utils.reduce_id_bits(canvas.segmentation)
    np.savez_compressed(FLAGS.seg_result_path+'seg_'+str(corner[0])+str(corner[1])+str(corner[2])+'_'+FLAGS.model_name, 
                        segmentation=seg, 
                        origins=canvas.origins)
    
    # save quantized probability map
    logging.info('saving probability map...')
    prob = canvas.seg_prob
    np.savez_compressed(FLAGS.seg_result_path+'prob_'+str(corner[0])+str(corner[1])+str(corner[2])+'_'+FLAGS.model_name, 
                        qprob=prob)

def run_inference(corner, subvol_size, model, image_path, seg_result_path):
    start = time()
    canvas = make_canvas(corner, subvol_size, model, image_path)
    canvas.segment_all()
    save_segmentation(corner, canvas, seg_result_path)
    total_time = int(time()-start)
    print('total time cost: ', total_time)
    return total_time

def main(argv):
    del argv
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
    model = load_network(basic_model.FFN(), model_path=FLAGS.base_dir+FLAGS.model_name+'.pkl')
    model.cuda()
    total_time = run_inference(FLAGS.corner,
                               FLAGS.subvol_size,
                               model,
                               FLAGS.image_path,
                               FLAGS.seg_result_path)
    os.system('python eval.py --model_name '+FLAGS.model_name+' --time '+str(total_time))

if __name__ == '__main__':
    flags.mark_flag_as_required('model_name')
    app.run(main)

#  python inference.py --model_name 3861925_1728000

