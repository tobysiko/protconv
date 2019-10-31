# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:35:42 2018

@author: ts149092
"""


#http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
import seaborn as sns
sns.set()

#rc('text', usetex=False)
plt.rcParams['figure.figsize']=(10,16)
import os,sys
import ProtConv2D.keras_cath_classifier_v3 as cs
from ProtConv2D.keras_cath_classifier_v3 import seq_from_onehot

ngpus = int(sys.argv[1])

# what-dimensional fingerprint? default=512D => last fully connected layer





root_path = sys.argv[2]
results_dir = sys.argv[3]



if ngpus==8:
    bs=512
elif ngpus==4:
    bs=64
else:
    bs=32
#bs=64*ngpus
 

fc1 = 0
fp_lengths =  [1024, 2048, 4096 ]
models_224 = ["fullconv"]

nchannels=3

dim=224   
fname = "CATH_20798-%i-%i-%i_2-3-4-5-6-7-8-9-10-11-12.hdf5"%(dim,dim,nchannels)
hname = os.path.join(root_path,fname)





for model in models_224:
    for l in fp_lengths:
        print (model,l)
        fc2 = l
        clf = cs.CATH_Classifier(
                        root_path, image_folder="pngbp", 
                        img_dim=dim, batch_size=bs, epochs=100, 
                        model=model, data_labels_filename="cath-domain-list.txt", 
                        label_columns="2,3,4,5,11", 
                        png_suffix="_rgb.png", nchannels=3, 
                        sample_size=None, selection_filename="cath-dataset-nonredundant-S40.txt", 
                        idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                        valsplit=0.4, save_resized_images=False, outdir=results_dir, 
                        use_pretrained=True, early_stopping="val_loss"#
                        , tensorboard=True, tbdir='C:/Users/ts149092/tblog',#'/local_scratch/ts149092/tblog',
                        optimizer="adam", verbose=True, batch_norm=True, act="relu",
                        learning_rate=-1, img_size_bins=[], dropout=0.25, img_bin_batch_sizes=[], 
                        flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                        fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=True,
                        generic_label="kcc-seq", h5_input=hname, h5_backend="h5py",
                        use_dc_decoder=False, dc_dec_weight=1.0, dc_dec_act="linear",
                        cath_loss_weight=0.1, dc_decoder_loss="mean_absolute_error",checkpoints=0,
                        cath_domlength_weight=0.01, domlength_regression_loss="mean_absolute_error",
                        use_seq_decoder=True, use_seq_encoder=True, seq_dec_weight=1.0, manual_seq_maxlength=1600,
                        seq_decoder_loss="categorical_crossentropy"
                        )

        modlabel = clf.train()
        for name in clf.history.history.keys():
            print ("%s: %.4f"%(name, clf.history.history[name][-1]))
        try:
            new_files =clf.visualize_image_data(use_pixels=False, use_pfp=True, save_path=results_dir, do_cluster_analysis=True, clustering_method="kmeans" )
        except Exception as e:
            print(e)
    

        for i in range(0, 8,2):
            print (i, i+1)
            s = np.array(clf.seqdata[i,:,:])[np.newaxis, ... ]

            print (s.shape)

            C,A,T,H,L,seq = clf.keras_model.predict([s], batch_size=1,verbose=1)
 
            
            seq_true = seq_from_onehot(clf.seqdata[i,:,:])
            seq_pred = seq_from_onehot(seq.reshape(1600,22))
            print (seq_true)
            print (seq_pred)
    

