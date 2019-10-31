# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:35:42 2018

@author: ts149092
"""


#http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html


import numpy as np

import os,sys
import ProtConv2D.keras_cath_classifier_v2 as cs


ngpus = int(sys.argv[1])
results_dir = sys.argv[2]

fc1 = 0
    

root_path ="/nasuni/projects/ts149092/cath"

if ngpus==8:
    bs=512
elif ngpus==4:
    bs=128
else:
    bs=64




#224x224 models

dim=224
nchannels=3

fname = "CATH_20798-%i-%i-%i_2-3-4-5-6-7-8-9-10-11-12.hdf5"%(dim,dim,nchannels)
hname = os.path.join("/nasuni/projects/ts149092/cath",fname)

fp_lengths =  [128, 512]
models_224 = ["densenet121", "vgg16", "resnet50"]



for model in models_224:
    for l in fp_lengths:
        print (model,l)
        fc2 = l
        clf = cs.CATH_Classifier(root_path, image_folder="pngbp", 
                                img_dim=dim, batch_size=bs, epochs=200, 
                                model=model, data_labels_filename="cath-domain-list.txt", 
                                label_columns="2,3,4,5,11", 
                                png_suffix="_rgb.png", nchannels=nchannels, 
                                sample_size=None, selection_filename="cath-domain-list.txt",#"cath-dataset-nonredundant-S40.txt", 
                                idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                                valsplit=0.4, save_resized_images=False, outdir=results_dir, 
                                use_pretrained=True, early_stopping="val_loss", tensorboard=False, 
                                optimizer="adam", verbose=True, batch_norm=True, act="relu",
                                learning_rate=-1, img_size_bins=[], dropout=0.25, img_bin_batch_sizes=[], 
                                flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                                fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                                generic_label="kcc_length", h5_input=hname, h5_backend="h5py",
                                cath_domlength_weight=1.0, domlength_regression_loss="mean_squared_logarithmic_error",
                                checkpoints=0

                                )

        clf.train()
        clf.visualize_image_data(use_pixels=False, use_pfp=True, save_path=results_dir, do_cluster_analysis=True, clustering_method="kmeans" )
    

