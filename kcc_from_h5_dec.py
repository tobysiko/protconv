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

model = sys.argv[1]
ngpus = int(sys.argv[2])

# what-dimensional fingerprint? default=512D => last fully connected layer
if len(sys.argv)==4:
    fc2 = int(sys.argv[3])
else:
    fc2 = 448

fc1 = 0
fc2 = 1024

root_path ="/nasuni/projects/ts149092/cath"
results_dir = os.path.join("results","cath_decoder")

if model in ["unet","lenet","lenet2","lenet3"]:
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
    bs=64
else:
    bs=32
#bs=64*ngpus
    
fname = "CATH-nonred-rgb-d%s.hdf5"%str(dim)


hname = os.path.join("/nasuni/projects/ts149092/",fname)

clf = cs.CATH_Classifier(root_path, image_folder="pngbp", 
                         img_dim=dim, batch_size=bs, epochs=200, 
                         model=model, data_labels_filename="cath-domain-list.txt", 
                         label_columns="2,3,4,5", 
                         png_suffix="_rgb.png", nchannels=3, 
                         sample_size=None, selection_filename="cath-dataset-nonredundant-S40.txt", 
                         idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                         valsplit=0.4, save_resized_images=False, outdir=results_dir, 
                         use_pretrained=True, early_stopping="val_loss", tensorboard=False, 
                         optimizer="adam", verbose=True, batch_norm=True, act="relu",
                         learning_rate=-1, img_size_bins=[], dropout=0.25, img_bin_batch_sizes=[], 
                         flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                         fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                         generic_label="kcc", h5_input=hname, h5_backend="h5py",
                         dc_decoder=True, dc_dec_weight=1.0, dc_dec_act="relu",
                         cath_loss_weight=0.5
                        )

modlabel = clf.train()
new_files =clf.visualize_image_data(use_pixels=False, use_pfp=True, save_path=results_dir, do_cluster_analysis=True, clustering_method="kmeans" )
for name in clf.history.history.keys():
    print ("%s: %.4f"%(name, clf.history.history[name][-1]))


for i in range(0, 16,4):
    print (i, i+1)
    
    x=np.squeeze(clf.X[i]).reshape((1,dim,dim,3)).astype('float32')
    print (x.shape)
    C,A,T,H,dec = clf.keras_model.predict([x])
    
    print( dec.reshape((dim,dim,3)).shape )
    
    
    plt.subplot(8,4,1 + i)
    im_o=plt.imshow(x.reshape(dim,dim,3),aspect='equal')
    plt.title("original")
    plt.colorbar(im_o)
    plt.subplot(8,4,2 + i)
    sns.distplot(x.flatten(),bins=256,kde=False, color='r').set_yscale('log')
    #histo.set_yscale('log');
    plt.xlabel("normalized values");plt.ylabel("log counts");plt.title("matrix histogram: original");plt.show()
    #plt.colorbar(im)   
    
    plt.subplot(8,4,3 + i)
    plt.imshow(dec.reshape(dim,dim,3), aspect='equal')
    plt.title("reconstruction")
    plt.colorbar()
    plt.subplot(8,4,4 + i)
    sns.distplot(dec.flatten(),bins=256,kde=False, color='r').set_yscale('log')
    #histo.set_yscale('log');
    plt.xlabel("normalized values");plt.ylabel("log counts");plt.title("matrix histogram: reconstructed");plt.show()

    plt.tight_layout()

plt.savefig(os.path.join(results_dir, "%s_image_grid.png"%modlabel))
    

#print "Reading...",hname,os.path.exists(hname)
#if True: #h5py
#    import h5py
#    hf = h5py.File(hname, "r")
#    
#    clf.X = np.array(hf["img"])
#    
#    clf.Ys = np.array(hf["class_labels"])
#    
#    clf.labels = list(hf["cath_codes"])
#    
#    clf.not_found = []
#    
#    print hf.keys(), clf.X.shape, len(clf.Ys), len(clf.labels)
#    
#    clf.train()
#    hf.close()
#    
#    
#else: #pytables
#    import tables
#    hf = tables.open_file(hname, mode='r')
#    hX = hf.root.img
#    hYs = hf.root.class_labels
#    hlabels = hf.root.cath_codes
#    clf.X = hX.read()
#    clf.Ys = hYs.read()
#    clf.labels = hlabels.read()
#    clf.not_found = []
#    
#    hf.close()
#    clf.train()

    

