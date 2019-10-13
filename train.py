# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:57:10 2019

@author: Ronglai Zuo
"""


import h5py, os, random
import numpy as np
import torch as t
from models import basic_model
from models import basic_model_11
from utils import utils
import torch.nn as nn

import logging
import itertools
import six
from time import time

from scipy.special import logit
from functools import partial

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('train_coords', 
                    'FIB-25/train_sample/coordinates_file.npy', 
                    'the file of coords')
flags.DEFINE_string('data_volumes',
                    'FIB-25/train_sample/grayscale_maps.h5:raw',
                    'the file of raw images')
flags.DEFINE_string('label_volumes',
                    'FIB-25/train_sample/groundtruth.h5:/volumes/labels/neuron_ids',
                    'groundtruth')
flags.DEFINE_string('checkpoints', 'checkpoints/originalFFN/basic11/', 'save checkpoints')

flags.DEFINE_integer('batch_size', 16, 'batchsize')
flags.DEFINE_integer('max_epochs', 1, 'epoch')
flags.DEFINE_integer('max_steps', 7000000, 'maxstep')
flags.DEFINE_list('fov_size', [33, 33, 33], 'fovsize_zyx')
flags.DEFINE_list('deltas', [8, 8, 8], 'deltas_zyx')

flags.DEFINE_float('threshold', 0.9, 'threshold')
flags.DEFINE_float('seed_pad', 0.05, 'seed_pad')
flags.DEFINE_integer('image_mean', 128, 'image_mean')
flags.DEFINE_integer('image_stddev', 33, 'image_stddev')

index = 0  #global, record the index of coordinates
max_index = 0

def load_network(network):
    logging.info('loading checkpoints...')
    save_path = '/mnt/dive/shared/ronglai.zuo/ffn-pytorch/checkpoints/originalFFN/basic11/2100164_1263000.pkl'
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


def get_one_input(coords, input_image_size, whole_images, whole_labels): 
    #whole_images, lables are numpy array
    #crop the image and labels, the center is coords
    global index
    input_image_radii = (input_image_size // 2).tolist() #(24,24,24),zyx
    input_image_size = input_image_size.tolist()  #zyx
    patch_shape = labels_shape = [1, 1] + input_image_size #(1,1,49,49,49)
    
    selector = []
    for l, h in zip(coords, input_image_radii):
        selector.append(slice(l-h, l+h+1))
    selector = tuple(selector)

    patch = whole_images[selector]
    patch = np.reshape(patch, patch_shape).astype(np.float32)
    
    labels = whole_labels[selector]
    labels = np.reshape(labels, labels_shape)

    patch -= 1.0*FLAGS.image_mean
    patch /= FLAGS.image_stddev

    lom = np.equal(labels, 
                   labels[0, 0, input_image_radii[0], input_image_radii[1], input_image_radii[2]]
                   ).astype(np.float32)
    labels = utils.soften_labels(lom)
    
    #augmentation--flip
    r = np.random.random()
    if r < 0.25:
        patch = np.flip(patch, 2)
        labels = np.flip(labels, 2)
    elif r < 0.5:
        patch = np.flip(patch, 3)
        labels = np.flip(labels, 3)
    elif r < 0.75:
        patch = np.flip(patch, 4)
        labels = np.flip(labels, 4)

    index += 1

    return patch, labels


def fixed_offsets(seed, fov_shifts):
    """Generates offsets based on a fixed list."""
    random.shuffle(fov_shifts)
    for off in itertools.chain([(0, 0, 0)], fov_shifts):
        is_valid_move = seed[:, 0,
                            seed.shape[2] // 2 + off[0],
                            seed.shape[3] // 2 + off[1],
                            seed.shape[4] // 2 + off[2]] >= logit(FLAGS.threshold)

        if not is_valid_move:
            continue

        yield off


def get_one_example(all_coords, input_size, get_offsets, whole_images, whole_labels, eval_tracker):
    
    while True:
        full_patch, full_labels = get_one_input(all_coords[index][::-1], input_size, whole_images, whole_labels)
        seed = logit(utils.initial_seed(input_size))

        for off in get_offsets(seed):
            pred_seed = utils.crop(seed, off, FLAGS.fov_size)
            patch = utils.crop(full_patch, off, FLAGS.fov_size)
            labels = utils.crop(full_labels, off, FLAGS.fov_size)
            
            assert pred_seed.base is seed
            yield pred_seed, patch, labels
            
        eval_tracker.eval_one_patch(full_labels, seed)
        

def get_batch(all_coords, input_size, batch_size, get_offsets, whole_images, whole_labels, eval_tracker):
    
    def _batch(iterable):
        for batch_vals in iterable:
            # `batch_vals` is sequence of `batch_size` tuples returned by the
            # `get_example` generator, to which we apply the following transformation:
            #   [(a0, b0), (a1, b1), .. (an, bn)] -> [(a0, a1, .., an),
            #                                         (b0, b1, .., bn)]
            # (where n is the batch size) to get a sequence, each element of which
            # represents a batch of values of a given type (e.g., seed, image, etc.)
            yield zip(*batch_vals)
    
    # Create a separate generator for every element in the batch. This generator
    # will automatically advance to a different training example once the allowed
    # moves for the current location are exhausted.
    for seeds, patch, labels in _batch(six.moves.zip(*[get_one_example(all_coords, 
                                                             input_size, 
                                                             get_offsets, 
                                                             whole_images, 
                                                             whole_labels, eval_tracker) for _ in range(batch_size)])):

        batched_seeds = np.concatenate(seeds)
        yield (batched_seeds, np.concatenate(patch), np.concatenate(labels))
        
        # batched_seed is updated in place with new predictions by the code
        # calling get_batch. Here we distribute these updated predictions back
        # to the buffer of every generator.
        for i in range(batch_size):
            seeds[i][:] = batched_seeds[i, ...]


def rand_eval(all_coords, model, whole_images, whole_labels, criterion):
    global index, max_index
    if index < max_index - 10:
        index_list = np.random.randint(low=index+1, high=max_index, size=(1,10))
    
    eval_loss = 0
    for i in range(10):
        patch, labels = get_one_input(all_coords[index_list[i]], 
                                      np.array(FLAGS.fov_size),
                                      whole_images,
                                      whole_labels)
        patch = patch.cuda()
        labels = labels.cuda()
        seed = logit(utils.initial_seed(FLAGS.fov_size))
        seed = seed.cuda()
        pred_seed = model(t.cat((patch, seed), 1))
        seed += pred_seed
        eval_loss += criterion(seed, labels)
    eval_loss /= 10
    return eval_loss.data().cpu()
    

def train_FFN(model):
    #multi-gpu on one machine, the max number of gpus is eight
    global max_index, index
    model = nn.DataParallel(model, device_ids=[0,1])
    model.cuda()
    fov_size = np.array(FLAGS.fov_size)
    deltas = np.array(FLAGS.deltas)
    input_image_size = fov_size + 2*deltas
    
    eval_tracker = utils.Eval_tracker(input_image_size, env='11')
    eval_tracker.reset()
    optimizer = t.optim.SGD(model.parameters(), lr = 0.001)
    criterion = basic_model.setup_loss()
    criterion = criterion.cuda()
    
    path, dataset = FLAGS.data_volumes.split(':')
    logging.info('reading images...')
    with h5py.File(path, 'r') as raw_images:
        whole_images = raw_images[dataset][:]
        
    path, dataset = FLAGS.label_volumes.split(':')
    logging.info('reading labels...')
    with h5py.File(path, 'r') as neuron_ids:
        whole_labels = neuron_ids[dataset][:]
    

    for epochs in range(FLAGS.max_epochs):
    
        logging.info('loading coordinates...')
        all_coords = np.load(FLAGS.train_coords)
        max_index = all_coords.shape[0]
        #print('the size of coords: ', all_coords.shape)  #(189941415, 3)
        fov_shifts = list(basic_model.fov_shifts(deltas=FLAGS.deltas))
        get_offsets = partial(fixed_offsets, fov_shifts = fov_shifts)
        
        the_batch = get_batch(all_coords, input_image_size, FLAGS.batch_size, get_offsets,
                              whole_images, whole_labels, eval_tracker)
        
        step = 0
        loss_100 = 0
        while step <= FLAGS.max_steps:

            if step % 100 == 0:
                start = time()

            seeds, patches, labels = next(the_batch)
            
            seeds_cuda = t.from_numpy(seeds).cuda()
            patches_cuda = t.from_numpy(patches).cuda()
            labels_cuda = t.from_numpy(labels).cuda()
            
            optimizer.zero_grad()
            pred_seeds = model(t.cat((patches_cuda, seeds_cuda), 1)) #logits 
            seeds_cuda += pred_seeds
            
            step += 1
            loss = criterion(seeds_cuda, labels_cuda)
            utils.update_seed(seeds, (0,0,0), seeds_cuda.detach().cpu().numpy())
            loss_100 += loss.data.cpu()
            
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                end = time()
                logging.info('sec per iter: %f, index: %d', (time()-start)/100, index)
                eval_tracker.plot(step, loss_100)
                eval_tracker.reset()
                logging.info('loss: %f', loss_100/100)
                loss_100 = 0
                
            if step % 500 == 0:
                logging.info('saving model...')
                t.save(model.state_dict(), FLAGS.checkpoints + str(index) + '_' + str(step) + '.pkl')
                logging.info('save success!')
                
    print('over!!!')
            

def main(argv):
    del argv
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    #model = basic_model_11.FFN()
    model = load_network(basic_model_11.FFN())
    train_FFN(model)
    
    
if __name__ == '__main__':
    app.run(main)



