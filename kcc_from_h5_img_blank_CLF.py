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
import pandas as pd
#rc('text', usetex=False)
plt.rcParams['figure.figsize']=(10,16)
import os,sys
import ProtConv2D.keras_cath_classifier_v3 as cs
from sklearn.metrics import mean_absolute_error
import numpy as np




if os.path.exists("C:/Users/ts149092/OneDrive - GSK/Documents/data"):
    root_path = os.path.abspath("C:/Users/ts149092/OneDrive - GSK/Documents/data")
    ngpus = 1
    results_dir = os.path.abspath("C:/Users/ts149092/OneDrive - GSK/Documents/results")
elif os.path.exists("/local_scratch/ts149092/data"):
    root_path = os.path.abspath("/local_scratch/ts149092/data")
    ngpus = 4
    results_dir = os.path.abspath("/hpc/ai_ml/ELT/results")
else:
    root_path = os.path.abspath("/home-us/ts149092/data")
    ngpus = 4
    results_dir = os.path.abspath("/hpc/ai_ml/ELT/results")

if ngpus==8:
    bs=512
elif ngpus==4:
    bs=128
else:
    bs=32
#bs=64*ngpus
 

fc1 = 0
fc2 = 128
param =  [0.0, 0.05, 0.1, 0.5, 0.95, 1.0]
model =  "fullconv"
nchannels=3
dim=256 
hname = os.path.join(root_path, "CATH_HEAVY_%ix%ix%i_n20798.hdf5"%(dim,dim,nchannels))
meta_file = os.path.join(root_path, "CATH_metadata.csv")





for blank_param in param:
    print (model, blank_param)
    
    clf = cs.CATH_Classifier(
                    root_path, 
                    img_dim=dim, batch_size=bs, epochs=100, 
                    model=model, data_labels_filename="cath-domain-list.txt", 
                    label_columns=['cath01_class', "cath02_architecture", "cath03_topology", "cath04_homologous_superfamily", "sequence_length"], 
                    png_suffix="_rgb.png", nchannels=3, 
                    sample_size=None, selection_filename="cath-dataset-nonredundant-S40.txt", 
                    
                    valsplit=0.3, save_resized_images=False, outdir=results_dir, 
                    
                    early_stopping="val_loss"#"none"#
                    , tensorboard=False, tbdir='/local_scratch/ts149092/tblog',#'/local_scratch/ts149092/tblog',
                    optimizer="adam", verbose=False, 
                    batch_norm_order="LAB",
                    act="relu",
                    
                    dropout=0.0,  
                    model_arch_file=None, model_weights_file=None,
                    fc1=fc1, fc2=fc2, keras_top=True ,ngpus=ngpus, train_verbose=False,
                    generic_label="blanks%s-lstm-clf"%(str(blank_param).replace(".","")), h5_backend="h5py",
                    
                    use_img_encoder=True, 
                    cath_loss_weight=1.0, dc_decoder_loss="mean_absolute_error",checkpoints=0,
                    cath_domlength_weight=0.1, domlength_regression_loss="mean_absolute_error",
                    blank_img_frac=blank_param,
                    
                    use_seq_encoder=True, use_seq_decoder=False, 
                    seq_dec_weight=1.0, manual_seq_maxlength=1400,
                    seq_decoder_loss="mean_absolute_error",
                    use_biovec=False, biovec_model="/nasuni/projects/ts149092/cath/cath_pdb_to_seq_protvec.model", generate_protvec_from_seqdict=False,
                    seq_from_pdb=False, pdb_folder="dompdb",
                    seq_enc_arch="lstm",
                    lstm_enc_length=1024,

                    conv_filter_1d=20, 
                    seq_encoding = "index",
                    no_classifier=False,
                    use_lstm_attention=False,
                    use_embedding=True,
                    classifier="dense",
                    seq_padding='pre',
                    fullconv_img_clf_length=0, fullconv_seq_clf_length=0,

                    merge_type='dot'
                    )


    clf.import_cath_dataset(img_h5_file=hname, metadata_file=meta_file)

    modlabel = clf.train_cath()

    
    for name in clf.history.history.keys():
        print ("%s: %.4f"%(name, clf.history.history[name][-1]))

    results = pd.DataFrame(data=clf.history.history)
    results.to_csv( os.path.join(results_dir, clf.modlabel+"_results-history.csv") )


    #try:
    if fc1==0 and fc2==0:
        pfp_layer = "final"
    else:
        pfp_layer = "fc_last"


    new_files =clf.visualize_cath_image_data(
            show_plots=False, 
            use_pixels=False, 
            use_pfp=True, 
            pfp_layername=pfp_layer, 
            ndims=2, 
            perp=100, early_exaggeration=4.0 ,learning_rate=100, angle=0.5, n_iter=1000, rseed=123, 
            pfp_length=fc2, 
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
    
    clf.plot_curves(metrics=['loss', 'cath01_class_loss'])
    
    clf.plot_curves(metrics=['cath01_class_acc', 'cath02_architecture_acc', 'cath03_topology_acc','cath04_homologous_superfamily_acc'])
    clf.plot_curves(metrics=['sequence_length_loss'])



            
    

