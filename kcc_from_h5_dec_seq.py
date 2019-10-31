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
#rc('text', usetex=False)
plt.rcParams['figure.figsize']=(10,16)
import os,sys
import ProtConv2D.keras_cath_classifier_v2 as cs
from ProtConv2D.keras_cath_classifier_v2 import seq_from_onehot


model = sys.argv[1]
ngpus = int(sys.argv[2])

# what-dimensional fingerprint? default=512D => last fully connected layer
if len(sys.argv)==4:
    fc2 = int(sys.argv[3])
else:
    fc2 = 448

fc1 = 512    

root_path ="/nasuni/projects/ts149092/cath"
results_dir = os.path.join("results","cath_decoder")

if model in ["lenet","lenet2","lenet3"]:
    dim = 224
elif model in ["vgg16","vgg19","resnet50","mobilenet","densenet121","densenet169","densenet201","nasnetmobile"]:
    dim = 224
elif model in ["inception","xception","inception_resnetv2"]:
    dim = 299
elif model in ["nasnetlarge"]:
    dim = 331
else:
    dim = None
    
if ngpus==8:
    bs=512
elif ngpus==4:
    bs=128
else:
    bs=64
#bs=64*ngpus

if dim%2==0:
    kernel_shape = "3,3"
else:
    kernel_shape = "4,4"
    
fname = "CATH-nonred-rgb-d%s.hdf5"%str(dim)
hname = os.path.join("/nasuni/projects/ts149092/",fname)

clf = cs.CATH_Classifier(root_path, image_folder="pngbp", 
                         img_dim=dim, batch_size=bs, epochs=200, 
                         model=model, data_labels_filename="cath-domain-list.txt", 
                         label_columns="2,3,4,5", 
                         png_suffix="_rgb.png", nchannels=3, 
                         sample_size=None, selection_filename="cath-dataset-nonredundant-S40.txt", 
                         idlength=7, 
                         kernel_shape=kernel_shape,
                         dim_ordering="channels_last", 
                         valsplit=0.4, save_resized_images=False, outdir="results", 
                         use_pretrained=True, early_stopping="val_loss", tensorboard=False, 
                         optimizer="adam", verbose=True, batch_norm=True, act="relu",
                         learning_rate=-1, img_size_bins=[], dropout=0.25, img_bin_batch_sizes=[], 
                         flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                         fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                         generic_label="kcc", h5_input=hname, h5_backend="h5py",
                         dc_decoder=True, dc_dec_weight=1.0, dc_dec_act="relu",
                         cath_loss_weight=0.5,
                         seq_encoder=True, seq_decoder=True, seq_dec_weight=0.1, seq_code_layer="fc_first",
                         seq_dec_act="relu", manual_seq_maxlength=1600
                        )


modlabel = clf.train()
newfiles = clf.visualize_image_data(use_pixels=False, use_pfp=True, save_path=results_dir, do_cluster_analysis=True, clustering_method="kmeans" )
for name in clf.history.history.keys():
    print ("%s: %.4f"%(name, clf.history.history[name][-1]))


    

for i in range(0, 8,2):
    print (i, i+1)
    x = np.squeeze(clf.X[i]).reshape((1,dim,dim,3))
    s = np.array(clf.seqdata[i,:,:])[np.newaxis, ... ]

    print (x.shape)
    print (s.shape)

    C,A,T,H,dec,seq = clf.keras_model.predict([x,s], batch_size=1,verbose=1)
    
    print( dec.reshape((dim,dim,3)).shape )


    plt.subplot(8,2,1 + i)
    plt.imshow(x.reshape(dim,dim,3),aspect='equal')
    #plt.colorbar(im)   
    
    plt.subplot(8,2,1 + i+1)
    plt.imshow(dec.reshape(dim,dim,3), aspect='equal')
    #plt.colorbar(im)   

    plt.tight_layout()
    
    
    seq_true = seq_from_onehot(clf.seqdata[i,:,:])
    seq_pred = seq_from_onehot(seq.reshape(1600,22))
    print (seq_true)
    print (seq_pred)
plt.savefig("%s_image_grid.png"%clf.modlabel)
    

