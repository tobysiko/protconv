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
import ProtConv2D.keras_cath_classifier_v2 as cs
from ProtConv2D.keras_cath_classifier_v2 import seq_from_onehot
ngpus = 1

# what-dimensional fingerprint? default=512D => last fully connected layer




base = sys.argv[1]

root_path ="/nasuni/projects/ts149092/cath"
results_dir = os.path.join("results","cath_decoder")


#base = "kcc_lenet2_pt1_224x224x3_20798_bt128_ep200_f3x3_fc512x400_cath2345_4G_split0.40_lo0.151_vlo4.330"
model_arch_file =os.path.join(results_dir, base+"_model_arch.json")
model_weights_file =os.path.join(results_dir, base+"_model_weights.hdf5")



nchannels=3
fc1=0
fc2=512
dim=224
ngpus=4
bs=64  
 

model = "fullconv"



  
fname = "CATH_20798-%i-%i-%i_2-3-4-5-6-7-8-9-10-11-12.hdf5"%(dim,dim,nchannels)
hname = os.path.join(root_path,fname)





clf = cs.CATH_Classifier(root_path, image_folder="pngbp", 
                         img_dim=dim, batch_size=bs, epochs=100, 
                         model=model, data_labels_filename="cath-domain-list.txt", 
                         label_columns="2,3,4,5,11", 
                         png_suffix="_rgb.png", nchannels=3, 
                         sample_size=None, selection_filename="cath-dataset-nonredundant-S40.txt", 
                         idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                         valsplit=0.4, save_resized_images=False, outdir=results_dir, 
                         use_pretrained=True, early_stopping="val_loss"#
                         , tensorboard=False, 
                         optimizer="adam", verbose=True, batch_norm=True, act="relu",
                         learning_rate=-1, img_size_bins=[], dropout=0.25, img_bin_batch_sizes=[], 
                         flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                         fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                         generic_label="kcc-dec", h5_input=hname, h5_backend="h5py",
                         dc_decoder=True, dc_dec_weight=1.0, dc_dec_act="linear",
                         cath_loss_weight=0.1, dc_decoder_loss="mean_absolute_error",checkpoints=0,
                         cath_domlength_weight=0.1, domlength_regression_loss="mean_squared_logarithmic_error",
			seq_encoder=True, seq_decoder=True, seq_dec_weight=1.0, seq_code_layer="fc_first",
                         seq_dec_act="relu", manual_seq_maxlength=1600
                        )



clf.prepare_dataset(show_classnames=False)
print (clf.keras_model)
clf.keras_model = clf.get_model()
print (clf.keras_model)
print (clf.keras_model.summary())

try:
    new_files =clf.visualize_image_data(use_pixels=False, use_pfp=True, save_path=results_dir, do_cluster_analysis=True, clustering_method="kmeans" )
except Exception as e:
    print(e)

for i in range(0, 16, 4):
    print (i, i+1)
    
    #x=np.squeeze(clf.X[i]).reshape((1,dim,dim,3))#.astype('float32')
    x = np.squeeze(clf.X[i]).reshape((1,dim,dim,3))
    s = np.array(clf.seqdata[i,:,:])[np.newaxis, ... ]

    print (x.shape)
    print (s.shape)


    print ("x.shape",x.shape, x.min(),x.max())
    C,A,T,H,L,dec,seq = clf.keras_model.predict([x,s], batch_size=1,verbose=1)

    

    #C,A,T,H,dec = clf.keras_model.predict([x])
    #L,dec = clf.keras_model.predict([x])

    #print(C.shape,A.shape,T.shape,H.shape,L.shape,dec.shape)

    print (dec.min(),dec.max())

    print("dec.reshape((dim,dim,3)).shape", dec.reshape((dim,dim,3)).shape )
    
    print("2D dist:",np.sqrt(np.mean((x-dec)**2)))
    
    plt.subplot(8,4,1 + i)
    im_o=plt.imshow(x.astype('int').reshape(dim,dim,3),aspect='equal')
    plt.title("original")
    #plt.colorbar(im_o)
    plt.subplot(8,4,2 + i)
    sns.distplot(x.flatten(),bins=256,kde=False, color='r')#.set_yscale('log')
    #histo.set_yscale('log');
    plt.xlabel("normalized values");plt.ylabel("counts");plt.title("matrix histogram: original");plt.show()
    #plt.colorbar(im)   
    
    plt.subplot(8,4,3 + i)
    im_d=plt.imshow(dec.astype('int').reshape(dim,dim,3), aspect='equal')
    plt.title("reconstruction")
    #plt.colorbar()
    plt.subplot(8,4,4 + i)
    sns.distplot(dec.flatten(),bins=256,kde=False, color='r')#.set_yscale('log')
    #histo.set_yscale('log');
    plt.xlabel("normalized values");plt.ylabel("counts");plt.title("matrix histogram: reconstructed");plt.show()

    plt.tight_layout()
    seq_true = seq_from_onehot(clf.seqdata[i,:,:])
    seq_pred = seq_from_onehot(seq.reshape(1600,22))
    print (C,A,T,H)
    print("Length predicted:",L)
    print([y[i] for y in clf.Ys])
    print("1D dist:",np.sqrt(np.mean( (clf.seqdata[i,:,:]-seq)**2)))
    print (i,"true seq",seq_true)
    plt.subplot(2,1,1)
    print(clf.seqdata[i,:,:].shape)
    im_seq_true = plt.imshow(clf.seqdata[i,:100,:].transpose().astype('float'),aspect='auto')
    plt.title(seq_true)
    plt.ylabels(list("-ACDEFGHIKLMNPQRSTVWYX"))
    print (i,"reconstructed",seq_pred)
    print(seq.shape)
    plt.subplot(2,1,2)
    im_seq_pred = plt.imshow(seq.reshape(1600,22).transpose().astype('float')[:100,:],aspect='auto')
    plt.title("reconstructed")

plt.savefig("%s_image_grid.png"%base)
    
