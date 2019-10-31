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
from sklearn.metrics import mean_absolute_error
import numpy as np





ngpus = int(sys.argv[1])
results_dir = sys.argv[2]



if ngpus==8:
    bs=512
elif ngpus==4:
    bs=128
else:
    bs=32
#bs=64*ngpus
 

fc1 = 0
fp_lengths =  [128]
models_224 = ["fullconv"]

nchannels=3

dim=256 
root_path =   "/home-us/ts149092/data/"
hname = os.path.join(root_path, "CATH_HEAVY_%ix%ix%i_n20798.hdf5"%(dim,dim,nchannels))

meta_file = os.path.join(root_path, "CATH_metadata.csv")




for model in models_224:
    for l in fp_lengths:
        print (model,l)
        fc2 = l
        clf = cs.CATH_Classifier(
                        root_path, image_folder="pngca", 
                        img_dim=dim, batch_size=bs, epochs=200, 
                        model=model, data_labels_filename="cath-domain-list.txt", 
                        label_columns=['cath01_class', "cath02_architecture", "cath03_topology", "cath04_homologous_superfamily", "sequence_length"], 
                        png_suffix="_rgb.png", nchannels=3, 
                        sample_size=None, selection_filename="cath-dataset-nonredundant-S40.txt", 
                        idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                        valsplit=0.3, save_resized_images=False, outdir=results_dir, 
                        valsplit_seed=None,
                        use_pretrained=True, early_stopping="val_loss"#"none"#
                        , tensorboard=False, tbdir='/local_scratch/ts149092/tblog',#'/local_scratch/ts149092/tblog',
                        optimizer="adam", verbose=False, 
                        batch_norm=True, batch_norm_order="LAB",
                        act="relu",
                        learning_rate=0.01, 
                        dropout=0.5, 
                        img_bin_batch_sizes=[], img_size_bins=[], 
                        flipdia_img=False, inverse_img=False, model_arch_file=None, model_weights_file=None,
                        fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=True,
                        generic_label="gru-seq-clf", h5_backend="h5py",
                        
                        use_img_encoder=False, use_img_decoder=False, 
                        dc_dec_weight=1.0, dc_dec_act="linear",
                        cath_loss_weight=1.0, dc_decoder_loss="mean_absolute_error",checkpoints=0,
                        cath_domlength_weight=0.5, domlength_regression_loss="mean_squared_logarithmic_error",
                        
                        
                        use_seq_encoder=True, use_seq_decoder=False, 
                        seq_dec_weight=1.0, manual_seq_maxlength=600,
                        seq_decoder_loss="mean_absolute_error",
                        use_biovec=False, biovec_model="/nasuni/projects/ts149092/cath/cath_pdb_to_seq_protvec.model",
                        seq_from_pdb=False, pdb_folder="dompdb",
                        seq_enc_arch="gru",seq_dec_arch="cnn", lstm_enc_length=64,lstm_dec_length=10,
                        generate_protvec_from_seqdict=False,
                        conv_filter_1d=5, 
                        seq_encoding = "index",
                        no_classifier=False,
                        use_lstm_attention=False,
                        fullconv_seq_clf_length=128,
                        use_embedding=True,
                        classifier="none"
                        )
        pfp_length = 512

        clf.import_cath_dataset(img_h5_file=hname, metadata_file=meta_file)

        modlabel = clf.train_cath()
        for name in clf.history.history.keys():
            print ("%s: %.4f"%(name, clf.history.history[name][-1]))
        #try:
        if clf.classifier!="dense":
            pfp_layer = "final"
        else:
            pfp_layer = "fc_last"
        
        clf.plot_curves(metrics=['loss', 'cath01_class_loss'])
        
        clf.plot_curves(metrics=['cath01_class_acc', 'cath02_architecture_acc', 'cath03_topology_acc','cath04_homologous_superfamily_acc'])
        clf.plot_curves(metrics=['sequence_length_loss'])
    
        new_files =clf.visualize_cath_image_data(
                show_plots=False, 
                use_pixels=False, 
                use_pfp=True, 
                pfp_layername=pfp_layer, 
                ndims=2, 
                perp=100, early_exaggeration=4.0 ,learning_rate=100, angle=0.5, n_iter=1000, rseed=123, 
                pfp_length=pfp_length, 
                marker_size=1, 
                n_jobs=1, 
                save_path=results_dir,
                do_cluster_analysis=False, 
                clustering_method="kmeans", 
                scores_dict={}
        )
        print(new_files)
        #except Exception as e:
        #    print("ERROR:", e)
        

        clf.plot_curves(metrics=['lr'])




            
    

