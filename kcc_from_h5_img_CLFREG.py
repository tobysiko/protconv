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
from sklearn.metrics import mean_absolute_error
import numpy as np





ngpus = int(sys.argv[1])
root_path = sys.argv[2]
results_dir = sys.argv[3]



if ngpus==8:
    bs=512
elif ngpus==4:
    bs=128
else:
    bs=32
#bs=64*ngpus
 

fc1 = 512
fp_lengths =  [512]
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
                        img_dim=dim, batch_size=bs, epochs=200, 
                        model=model, data_labels_filename="cath-domain-list.txt", 
                        label_columns="2,3,4,5,11", 
                        png_suffix="_rgb.png", nchannels=3, 
                        sample_size=None, selection_filename="cath-dataset-nonredundant-S40.txt", 
                        idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                        valsplit=0.4, save_resized_images=False, outdir=results_dir, 
                        use_pretrained=True, early_stopping="val_loss"#"none"#
                        , tensorboard=True, tbdir='/local-scratch/ts149092/tblog',#'/local_scratch/ts149092/tblog',
                        optimizer="adam", verbose=True, 
                        batch_norm=True, batch_norm_order="BLA",
                        act="relu",
                        learning_rate=-1, img_size_bins=[], 
                        dropout=0.00, img_bin_batch_sizes=[], 
                        flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                        fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                        generic_label="img-clf", h5_input=hname, h5_backend="h5py", checkpoints=0,
                        cath_loss_weight=1.0, 
                        cath_domlength_weight=1.0, domlength_regression_loss="mean_absolute_error",

                        use_img_encoder=True, use_img_decoder=False, 
                        dc_dec_weight=1.0, dc_dec_act="linear",
                        dc_decoder_loss="mean_absolute_error",
                        
                        
                        use_seq_encoder=False, use_seq_decoder=False,  
                        seq_dec_weight=0.5, manual_seq_maxlength=1600,
                        seq_decoder_loss="mean_absolute_error",
                        use_biovec=False, biovec_model="/nasuni/projects/ts149092/cath/cath_pdb_to_seq_protvec.model",
                        seq_from_pdb=True, pdb_folder="dompdb",
                        generate_protvec_from_seqdict=False
                        )

        modlabel = clf.train()
        for name in clf.history.history.keys():
            print ("%s: %.4f"%(name, clf.history.history[name][-1]))
        #try:
            
        new_files =clf.visualize_image_data(
                show_plots=False, 
                use_pixels=False, 
                use_pfp=True, 
                pfp_layername="fc_last", 
                ndims=2, 
                perp=100, early_exaggeration=4.0 ,learning_rate=100, angle=0.5, n_iter=1000, rseed=123, 
                pfp_length=512, 
                marker_size=1, 
                n_jobs=1, 
                save_path=results_dir,
                do_cluster_analysis=True, 
                clustering_method="kmeans", 
                scores_dict={}
        )
        print(new_files)
        #except Exception as e:
        #    print(e)
        
        clf.plot_curves(metrics=['loss', 'Cout_loss','Hout_loss'])
        
        clf.plot_curves(metrics=['Cout_acc', 'Aout_acc', 'Tout_acc','Hout_acc'])


        clf.plot_curves(metrics=['LENout'])

        for i in [1,2,10,20,100,200,1000,2000,10000,20000]:
            print (i)
            x=np.squeeze(clf.X[i]).reshape((1,dim,dim,3))#.astype('float32')
            print (x.shape)
            C,A,T,H,L = clf.keras_model.predict([x], batch_size=1,verbose=1)


            print (C,A,T,H)
            print("Length predicted:",L)
            print([y[i] for y in clf.Ys])

            
    

