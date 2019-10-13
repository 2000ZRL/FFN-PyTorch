#!/usr/bin/python

from cremi.io.CremiFile import *
from cremi.evaluation.NeuronIds import *
from cremi.evaluation.Clefts import *
from cremi.evaluation.SynapticPartners import *
import numpy as np
import h5py
from math import sqrt
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('time', None, 'time')
flags.DEFINE_string('base_dir', 'results/test_sample/basic11/', 'seg_result_base_dir')
flags.DEFINE_string('model_name', None, 'model_name')
flags.DEFINE_string('results_file', 'results/test_sample/ckpsel11.txt', 'results_file')

def main(argv):
    del argv
    npz_file = FLAGS.base_dir+'seg_000_'+FLAGS.model_name+'.npz'
    hdf_file = FLAGS.base_dir+'seg_000_'+FLAGS.model_name+'.hdf'
    gt_file ='FIB-25/test_sample/groundtruth.h5'
    pred = np.load(npz_file)
    img=pred['segmentation']
    print(np.unique(img))
    print(img.shape)

    with h5py.File(hdf_file,'w') as f:
        neuron_ids = f.create_dataset('volumes/labels/neuron_ids',data = img)

    with h5py.File(hdf_file,'r+') as f1:
        f1['volumes/labels/neuron_ids'].attrs.create('resolution',[8.0,8.0,8.0])

    test = CremiFile(hdf_file, 'r')
    truth = CremiFile(gt_file, 'r')

    neuron_ids_evaluation = NeuronIds(truth.read_neuron_ids())

    (voi_split, voi_merge) = neuron_ids_evaluation.voi(test.read_neuron_ids())
    adapted_rand = neuron_ids_evaluation.adapted_rand(test.read_neuron_ids())

    print ("Neuron IDs")
    print ("==========")
    print ("\tvoi split   : " + str(voi_split))
    print ("\tvoi merge   : " + str(voi_merge))
    voi_total = voi_split + voi_merge
    print ("\tvoi total   : " + str(voi_total))
    print ("\tadapted RAND: " + str(adapted_rand))
    cremi_score = sqrt(voi_total*adapted_rand)
    print ("\tcremi score : " + str(cremi_score))
    print ("\tmodel name  : " + FLAGS.model_name)
    
    with open(FLAGS.results_file, 'a+') as f:
        f.write("\nvoi split   : " + str(voi_split)+"\nvoi merge   : " + str(voi_merge)+\
                "\nvoi total   : " + str(voi_total)+"\nadapted RAND: " + str(adapted_rand)+\
                "\ncremi score : " + str(cremi_score)+\
                "\ntime cost   : " + FLAGS.time+'\n\n')

if __name__ == '__main__':
    flags.mark_flag_as_required('model_name')
    flags.mark_flag_as_required('time')
    app.run(main)
'''
clefts_evaluation = Clefts(test.read_clefts(), truth.read_clefts())

false_positive_count = clefts_evaluation.count_false_positives()
false_negative_count = clefts_evaluation.count_false_negatives()

false_positive_stats = clefts_evaluation.acc_false_positives()
false_negative_stats = clefts_evaluation.acc_false_negatives()

print ("Clefts")
print ("======")

print ("\tfalse positives: " + str(false_positive_count))
print ("\tfalse negatives: " + str(false_negative_count))

print ("\tdistance to ground truth: " + str(false_positive_stats))
print ("\tdistance to proposal    : " + str(false_negative_stats))

synaptic_partners_evaluation = SynapticPartners()
fscore = synaptic_partners_evaluation.fscore(test.read_annotations(), truth.read_annotations(), truth.read_neuron_ids())

print ("Synaptic partners")
print ("=================")
print ("\tfscore: " + str(fscore))
'''
