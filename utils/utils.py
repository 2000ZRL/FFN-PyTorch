# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:33:23 2019

@author: Ronglai Zuo
"""
#utils
import numpy as np
import visdom
from scipy.special import logit
from scipy.special import expit
from collections import deque
from queue import Queue

def soften_labels(lom, softness=0.05):
    
    return np.clip(lom, softness, 1-softness).astype(np.float32)


def crop(data, offset, crop_shape):
    #tensor [batch,channel,x,y,z] 49,49,49  crop_shape 33,33,33
    shape = np.array(data.shape[2:])
    crop_shape = np.array(crop_shape)
    offset = np.array(offset)
    start = shape // 2 - crop_shape // 2 + offset
    end = start + crop_shape
    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] + [slice(None)] + selector)
    
    cropped = data[selector]
    return cropped


def initial_seed(shape, pad=0.05, seed=0.95):

    seed_array = np.full(([1,1] + list(shape)), pad, dtype=np.float32)
    idx = tuple([slice(None), slice(None)] + list(np.array(shape) // 2))
    seed_array[idx] = seed #center point = 0.95
    return seed_array


def update_seed(to_update, offset, new_value):
    
    shape = np.array(to_update.shape[2:])
    crop_shape = np.array(new_value.shape[2:])
    offset = np.array(offset)

    start = shape // 2 - crop_shape // 2 + offset
    end = start + crop_shape

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None), slice(None)] + selector)
    
    to_update[selector] = new_value


class Eval_tracker(object):
    
    def __init__(self, shape, port='8097', env='main'):
        self.reset()
        self.eval_threshold = logit(0.9)
        self._eval_shape = shape
        self.vis = visdom.Visdom(port=port, env=env)
        self.win_loss = self.vis.line(X = np.array([0]), 
                                      Y = np.array([0]), 
                                      opts = dict(title = 'loss'))
        self.win_precision = self.vis.line(X = np.array([0]), 
                                           Y = np.array([0]), 
                                           opts = dict(title = 'precision'))
        self.win_recall = self.vis.line(X = np.array([0]), 
                                        Y = np.array([0]), 
                                        opts = dict(title = 'recall'))
        self.win_accuracy = self.vis.line(X = np.array([0]), 
                                          Y = np.array([0]), 
                                          opts = dict(title = 'accuracy'))
        self.win_f1 = self.vis.line(X = np.array([0]), 
                                    Y = np.array([0]), 
                                    opts = dict(title = 'f1'))
        self.win_images_xy = self.vis.images(np.random.randn(10,1,49,98), 
                                             opts = dict(title = 'images_xy'))
        
        
    def reset(self):
        #self.num_patches = 0
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.total_voxels = 0
        self.masked_voxels = 0
        self.images_xy = Queue(maxsize=10)
        self.images_yz = []
        self.images_xz = []
        
    
    def eval_one_patch(self, labels, predicted):
        #pred = crop(predicted, (0,0,0), self._eval_shape)
        #labels = crop(labels, (0,0,0), self._eval_shape)
        #loss = F.binary_cross_entropy_with_logits(t.from_numpy(pred), t.from_numpy(labels))
        
        #self.total_voxels += labels.size
        labels = crop(labels, (0,0,0), self._eval_shape)
        predicted = crop(predicted, (0,0,0), self._eval_shape)
        
        pred_mask = predicted >= self.eval_threshold
        true_mask = labels > 0.5
        pred_bg = np.logical_not(pred_mask)
        true_bg = np.logical_not(true_mask)
        
        self.tp += np.sum(pred_mask & true_mask)
        self.fp += np.sum(pred_mask & true_bg)
        self.fn += np.sum(pred_bg & true_mask)
        self.tn += np.sum(pred_bg & true_bg)
        #self.num_patches += 1
        
        selector = [slice(None), slice(None), labels.shape[2]//2, slice(None), slice(None)]
        la = (labels[tuple(selector)] * 255).astype(np.int8)
        pred = expit(predicted)
        pred = (pred[tuple(selector)] * 255).astype(np.int8)
        if not self.images_xy.full():
            self.images_xy.put(np.concatenate((la, pred), axis=3))
        
        
    def plot(self, step, loss):
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        accuracy = (self.tp + self.tn) / max(self.tp + self.tn + self.fn + self.fp, 1)
        f1 = (2.0 * precision * recall)/max(precision + recall, 1)
        self.vis.line(X = np.array([step]), 
                      Y = np.array([precision]), 
                      win=self.win_precision, 
                      update='append')       
        
        self.vis.line(X = np.array([step]), 
                      Y = np.array([recall]), 
                      win=self.win_recall, 
                      update='append')
        
        self.vis.line(X = np.array([step]), 
                      Y = np.array([accuracy]), 
                      win=self.win_accuracy, 
                      update='append')
        
        self.vis.line(X = np.array([step]), 
                      Y = np.array([f1]), 
                      win=self.win_f1, 
                      update='append')
        
        self.vis.line(X = np.array([step]), 
                      Y = np.array([loss/100]), 
                      win=self.win_loss, 
                      update='append')
        '''
        self.vis.line(X = np.array([step]), 
                      Y = np.array([eval_loss]), 
                      win=self.win_loss, 
                      update='append')
        '''
        pred_and_labels = self.images_xy.get()
        while not self.images_xy.empty():
            pred_and_labels = np.concatenate((pred_and_labels, self.images_xy.get()))
            
        self.vis.images(pred_and_labels, nrow=5, 
                        padding=2, 
                        win=self.win_images_xy)
        
    def plot_loss(self, step, loss):
        self.vis.line(X = np.array([step]), 
                      Y = np.array([loss/100]), 
                      win=self.win_loss, 
                      update='append')
        
'''
class FIB25(data.Dataset):
    
    def __init__(self, input_image_size, fov_size, batch_size, coords_file, image_file, label_file, 
                 fov_shifts, threshold, image_mean, image_stddev):
        
        logging.info('loading coords...')
        all_coords = np.load(coords_file)
        self.all_coords = all_coords
        
        path, dataset = image_file.split(':')
        logging.info('reading images...')
        with h5py.File(path, 'r') as raw_images:
            self.whole_images = raw_images[dataset][:]
        
        path, dataset = label_file.split(':')
        logging.info('reading labels...')
        with h5py.File(path, 'r') as neuron_ids:
            self.whole_labels = neuron_ids[dataset][:]
        
        self.fov_shifts = fov_shifts
        self.input_image_size = input_image_size
        self.input_image_radii = self.input_image_size // 2
        self.fov_size = fov_size.tolist()
        self.batch_size = batch_size
        self.threshold = threshold
        self.image_mean = image_mean
        self.image_stddev = image_stddev
        
        logging.info('initializing...')
        seeds_list = []  #global
        for _ in range(self.batch_size):
            seeds_list.append(logit(initial_seed(self.input_image_size)))
        self.seeds_list = seeds_list
        
        fov_list = []  #global
        for _ in range(self.batch_size):
            fov_list.append(self.fov_shifts)
        self.fov_list = fov_list
    
        
    def fixed_offsets(self, seed):
        """Generates offsets based on a fixed list."""
        for off in itertools.chain([(0, 0, 0)], self.fov_shifts):
            is_valid_move = seed[0,
                                seed.shape[1] // 2 + off[0],
                                seed.shape[2] // 2 + off[1],
                                seed.shape[3] // 2 + off[2]] >= logit(self.threshold)

            if not is_valid_move:
                continue

            yield off
            
            
    def is_valid_off(self, seed_index, off):
        shape = self.seed_list[seed_index].shape
        return self.seed_list[seed_index][0, 
                             shape[1]//2+off[0], 
                             shape[2]//2+off[1], 
                             shape[3]//2+off[2]] >= logit(self.threshold)
        
    
    def get_one_input(self, index):
        coords = self.all_coords[index]
        input_image_radii = self.input_image_radii.tolist()  #24,24,24
        input_image_size = self.input_image_size.tolist()  #49,49,49
        patch_shape = labels_shape = [1] + input_image_size[::]
        
        selector = []
        for l, h in zip(coords, input_image_radii):
            selector.append(slice(l-h, l+h+1))
        selector = tuple(selector)
        
        patch = self.whole_images[selector]
        patch = np.reshape(patch, patch_shape).astype(np.float32)
    
        labels = self.whole_labels[selector]
        labels = np.reshape(labels, labels_shape)
    
        patch -= 1.0 * self.image_mean
        patch /= self.image_stddev
    
        lom = np.equal(labels, 
                       labels[0, input_image_radii[0], input_image_radii[1], input_image_radii[2]]
                       ).astype(np.float32)
        labels = soften_labels(lom)
        #logging.info('get one input success!')
        return patch, labels, coords
    
    
    
    def get_one_example(self, index):
        seed_index = index % self.batch_size
        full_patch, full_labels, coords = self.get_one_input(index)
        #seed = logit(initial_seed(self.input_image_size))
            
        if(self.fov_list[sed_index] != []):
            off = self.fov_list[seed_index].pop(0)
            while not self.is_valid_off(seed_index, off):
                off = self.fov_list[seed_index].pop(0)
        
        
        print('offsets: ', off)
        pred_seed = crop(self.seeds_list[seed_index], off, self.fov_size)
        patch = crop(full_patch, off, self.fov_size)
        label = crop(full_labels, off, self.fov_size)
                
        assert pred_seed.base is seed
        yield pred_seed, patch, label
                


    def __getitem__(self, index):
        for pred_seed, patch, label in self.get_one_example(index):
            
            return pred_seed[:], patch[:], label[:]
        
        
    def __len__(self):
        return len(self.all_coords)

'''
