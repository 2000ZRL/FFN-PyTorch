# FFN-PyTorch
Reimplementation of Flood Filling Network based on Pytorch

In the summer of 2019, I was conducting summer research for three months in [Prof. Shuiwang Ji](http://people.tamu.edu/~sji/)'s lab, Texas A&M University. 
In the first month I finished reimplementing Flood-Filling Network by PyTorch. 
I also modify my codes to run FFN on multi-gpus (eight gpus at most) by DataParallelism. 
In addition, I wrote eval.py to evaluate the segmentation results automatically. And evaluation metrics can be seen [here](https://github.com/cremi/cremi_python). 
In order to select the best model on validation dataset, I also wrote ckp_sel.py to do the job automatically.

How to get data? Please read README.md of [original FFN](https://github.com/google/ffn) proposed by Google AI. The data are stored in their Google Cloud.

How to put the data? After you get the training and test dataset, please put them in FIB-25/train_sample and FIB-25/test_sample, respectively
If you put them somewhere else, you might need to modify the 'flags' in my scripts.

How to run the scripts? 
Before you train the model, please run partition.py and then build_coordinates.py in 'data'. What these two scripts do can be found in README.md of [original FFN](https://github.com/google/ffn).
Run train.py. Before that please set Visdom on your server. One checkpoint is stored in 'checkpoints', although it might not be the best one.
Run inference.py
Run eval.py
