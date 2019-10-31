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

# what-dimensional fingerprint? default=512D => last fully connected layer
fc1 = 0
    

root_path ="/nasuni/projects/ts149092/cath"


# if model in ["lenet","lenet2","lenet3"]:
#     dim = 128
# elif model in ["vgg16","vgg19","resnet50","mobilenet","densenet121","densenet169","densenet201"]:
#     dim = 224
# elif model in ["inception","xception","inception_resnetv2"]:
#     dim = 299
# else:
#     dim = None

if ngpus==8:
    bs=512
elif ngpus==4:
    bs=128
else:
    bs=64
#bs=64*ngpus


fp_lengths =  [128, 512]


#224x224 models
dim=224
fname = "CATH_20798-%i-%i-3.hdf5"%(dim,dim)
hname = os.path.join(root_path, fname)


models_224 = ["densenet121", "vgg16", "resnet50"]



for model in models_224:
    for l in fp_lengths:
        print (model,l)
        fc2 = l
        clf = cs.CATH_Classifier(root_path, image_folder="pngbp", 
                                img_dim=dim, batch_size=bs, epochs=200, 
                                model=model, data_labels_filename="cath-domain-list.txt", 
                                label_columns="2,3,4,5", 
                                png_suffix="_rgb.png", nchannels=3, 
                                sample_size=None, selection_filename="cath-domain-list.txt",#"cath-dataset-nonredundant-S40.txt", 
                                idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                                valsplit=0.4, save_resized_images=False, outdir=results_dir, 
                                use_pretrained=True, early_stopping="val_loss", tensorboard=False, 
                                optimizer="adam", verbose=True, batch_norm=True, act="relu",
                                learning_rate=-1, img_size_bins=[], dropout=0.25, img_bin_batch_sizes=[], 
                                flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                                fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                                generic_label="kcc", h5_input=hname, h5_backend="h5py",checkpoints=0
                                )

        clf.train()
        clf.visualize_image_data(use_pixels=False, use_pfp=True, save_path=results_dir, do_cluster_analysis=True, clustering_method="kmeans" )
    

#299x299 models
dim=299
fname = "CATH_20798-%i-%i-3.hdf5"%(dim,dim)
hname = os.path.join(root_path, fname)


models_299 = ["inception","inception_resnetv2"]
for model in models_299:
    for l in fp_lengths:
        print(model,l)
        fc2 = l
        clf = cs.CATH_Classifier(root_path, image_folder="pngbp", 
                                img_dim=dim, batch_size=bs, epochs=200, 
                                model=model, data_labels_filename="cath-domain-list.txt", 
                                label_columns="2,3,4,5", 
                                png_suffix="_rgb.png", nchannels=3, 
                                sample_size=None, selection_filename="cath-domain-list.txt",#"cath-dataset-nonredundant-S40.txt", 
                                idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                                valsplit=0.4, save_resized_images=False, outdir=results_dir, 
                                use_pretrained=True, early_stopping="val_loss", tensorboard=False, 
                                optimizer="adam", verbose=True, batch_norm=True, act="relu",
                                learning_rate=-1, img_size_bins=[], dropout=0.25, img_bin_batch_sizes=[], 
                                flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                                fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                                generic_label="kcc", h5_input=hname, h5_backend="h5py",checkpoints=0
                                )

        clf.train()
        clf.visualize_image_data(use_pixels=False, use_pfp=True, save_path=results_dir, do_cluster_analysis=True, clustering_method="kmeans" )
#299x299 models
dim=331
fname = "CATH_20798-%i-%i-3.hdf5"%(dim,dim)
hname = os.path.join(root_path, fname)


models_331 = ["nasnetlarge"]
for model in models_331:
    for l in fp_lengths:
        print(model,l)
        fc2 = l
        clf = cs.CATH_Classifier(root_path, image_folder="pngbp", 
                                img_dim=dim, batch_size=bs, epochs=200, 
                                model=model, data_labels_filename="cath-domain-list.txt", 
                                label_columns="2,3,4,5", 
                                png_suffix="_rgb.png", nchannels=3, 
                                sample_size=None, selection_filename="cath-domain-list.txt",#"cath-dataset-nonredundant-S40.txt", 
                                idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                                valsplit=0.4, save_resized_images=False, outdir=results_dir, 
                                use_pretrained=True, early_stopping="val_loss", tensorboard=False, 
                                optimizer="adam", verbose=True, batch_norm=True, act="relu",
                                learning_rate=-1, img_size_bins=[], dropout=0.25, img_bin_batch_sizes=[], 
                                flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                                fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                                generic_label="kcc", h5_input=hname, h5_backend="h5py",checkpoints=0
                                )

        clf.train()
        clf.visualize_image_data(use_pixels=False, use_pfp=True, save_path=results_dir, do_cluster_analysis=True, clustering_method="kmeans" )