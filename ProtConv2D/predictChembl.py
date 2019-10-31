# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:39:58 2017

@author: ts149092
"""
#from prot2image import pdb2mat, plotMatrix
import pandas as pd
import numpy as np
import os, glob, ast
import urllib
from ProtConv2D.MLhelpers import addOnData, DFcounts, scaleColumns, compareDimRed, perform_RF, perform_DNN, shuffleDFrows, perform_rnn,print_progress,add_compound_fingerprints,plot_confusion_matrix, plot_hist, plot_ROC
#from pdbfixer import PDBFixer
#from simtk.openmm.app import PDBFile
#
from collections import defaultdict
import matplotlib.pylab as plt
#%matplotlib inline
import matplotlib.cm as cm

from Bio.PDB import PDBParser
from keras.models import model_from_json, Sequential, Model
from keras.layers import Dense, Activation, Dropout, GlobalAveragePooling2D,GlobalAveragePooling1D,GlobalAveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K
from scipy.misc import imread,imresize
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc, recall_score, accuracy_score, confusion_matrix, silhouette_score, precision_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold

import ProtConv2D.pdb2image_v2
import json
import logging, sys


def get_PDBe_API_data(api_url) :
    num_trials = 3
    for trial_number in range(num_trials) :
        try :
            return json.loads(urllib.urlopen(api_url).read())
        except Exception as err : # OLD: except Exception, err :
            logging.warn("Error fetching PDBe-API data! Trial number %d for call %s" % (trial_number, api_url))
            if trial_number == num_trials-1 :
                raise err

def setup_PDBe_access():
    
    # configure the logger
    # btw, reload is just a hack to make logging work in the notebook, it's usually uncessary
    reload(logging)
    logging.basicConfig(
        level=logging.DEBUG, stream=sys.stdout,
        format='LOG|%(asctime)s|%(levelname)s  %(message)s', datefmt='%d-%b-%Y %H:%M:%S'
    )
    

def fetch_info_from_PDB(pdb_ids_list):
    
    PDBE_API_URL="http://www.ebi.ac.uk/pdbe/api"
    print( pdb_ids_list)
    for pdb_id in pdb_ids_list:
        pub_url = PDBE_API_URL + "/pdb/entry/summary/" +pdb_id
        try:
            data = get_PDBe_API_data(pub_url)[pdb_id]
        except:
            logging.warn("Entry publications could not be obtained for PDB id " + pdb_id)
        else:
            print( data)



def mapUniprot2PDB(idlist):
    #http://www.uniprot.org/help/api_idmapping
    url = 'http://www.uniprot.org/uploadlists/'
    
    params = {
    'from':'ACC',
    'to':'PDB_ID',
    'format':'tab',
    'query':'%s'%" ".join(idlist)
    }
    
    data = urllib.urlencode(params)
    request = urllib.Request(url, data)
    contact = "" # Please set your email address here to help us debug in case of problems.
    request.add_header('User-Agent', 'Python %s' % contact)
    response = urllib.urlopen(request)
    page = response.read(200000)
    
    return page

def fetchRedoPDB(pdb_id, folder):
    # see: http://www.cmbi.ru.nl/pdb_redo/downloads.html
    p = pdb_id.lower()
    
    url = "http://www.cmbi.ru.nl/pdb_redo/%s/%s/%s_final.pdb"%(p[1:3], p, p)
    #print url
    try:
        local = os.path.join(folder, "%s.pdb"%p)
        urllib.urlretrieve(url, local)
        
    except Exception as e:
        print( p, e )
        return False
    
    if "404 Not Found" in open(local).read():
        os.remove(local)
        return False
    return os.path.exists(local)

def fetchRCSBPDB(pdb_id, folder):
    p = pdb_id.upper()
    
    url = "http://files.rcsb.org/download/%s.pdb"%(p)
    #print url
    try:
        local = os.path.join(folder, "%s.pdb"%p.lower())
        urllib.urlretrieve(url, local)
        
    except Exception as e:
        print( p, e )
        return False
    
    return os.path.exists(local)

def download_representative_pdbs(pdblist, pdbfolder):
    print( pdbfolder )
    print_progress(0, len(pdblist), prefix='Downloading from REDO_PDB', suffix=' Success', decimals=1, bar_length=50)
    
    for pi in xrange(len(pdblist)):
        p = pdblist[pi]
        print_progress(pi+1, len(pdblist), prefix='Downloading from REDO_PDB', suffix=' Success', decimals=1, bar_length=50)
        
        representative = None
        # pick first structure in list that is available from REDO
        found_structure = False
        for pdb in p:
            if not os.path.exists(os.path.join(pdbfolder, "%s.pdb"%pdb.lower())):
                if fetchRedoPDB(pdb, pdbfolder):
                    found_structure = True
                    representative = pdb
                    break
        # in case none of the listed structures is available in REDO, fetch from RCSB
        if not found_structure:
            for pdb in p:
                if not os.path.exists(os.path.join(pdbfolder, "%s.pdb"%pdb.lower())):
                    if fetchRCSBPDB(pdb, pdbfolder):
                        found_structure = True
                        representative = pdb
                        break
        pdblist[pi] = [representative]

def fix_and_clean_pdb(pdblist, pdbfolder): # DOES THIS WORK?
    import pdbfix
    for pi in xrange(len(pdblist)):
        p = pdblist[pi]
        #CONTINUE
        for pdb in p:
            infile = os.path.join(pdbfolder, pdb.lower()+".pdb")
            outfile = os.path.join(pdbfixedfolder, pdb.lower()+".pdb")
            if os.path.exists(infile) and not os.path.exists(outfile):
                pdbfix.fixPDB(infile, outfile, method="BioPython", PERM=1, QUIET=True, atomchecker=None)
def convert_pdb_to_image(domlist, basefolder, pdbfolder, pngfolder, img_dim, overwrite_files=False ,show_pairplot=False, show_distplots=False, show_matrices=False, show_protein=False):
    from pdb2image import PDBToImageConverter
    import prody
    prody.confProDy(verbosity="critical") #(‘critical’, ‘debug’, ‘error’, ‘info’, ‘none’, or ‘warning’)
    
    # initialize object with all arguments
    protein2image_converter = PDBToImageConverter(
                    basefolder, 
                    os.path.join(basefolder,pdbfolder) , 
                    os.path.join(basefolder,pngfolder), 
                    os.path.join(basefolder,"npy"), 
                    domlist,
                    img_dim=img_dim, 
                    overwrite_files=overwrite_files,
                    selection="protein and heavy",
                    pdb_suffix=".pqr",
                    channels=["dist","anm","electro"],
                    
                    maxnatoms=10000
                 )
    # convert pdb to image
    print( "converting proteins to images...", pngfolder)
    protein2image_converter.convert_pdbs_to_arrays(show_pairplot=show_pairplot, show_distplots=show_distplots, show_matrices=show_matrices, show_protein=show_protein)
    
    
def fingerprint_distance_matrix(fprint_matrix, labels=None, metric="euclidean",n_jobs=1):
    if not metric in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]:
        metric = "euclidean"
    from sklearn.metrics.pairwise import pairwise_distances
    dmat = pairwise_distances(fprint_matrix, Y=None, metric=metric, n_jobs=n_jobs)
    
    return dmat

    
def create_binary_class_DNN(input_dim, arch=[200,150,100,50,25], act ="relu", dropout=0.1, my_metrics=['accuracy'], lr=0.001, decay=0.0):
    m = Sequential()
    for i in xrange(len(arch)):
        print( "hidden layer %i (%i neurons)"%(i,arch[i]) )
        if i==0:
            m.add(Dense(arch[0], input_dim=input_dim))
        else:
            m.add(Dense(arch[i]))
        m.add(BatchNormalization())
        m.add(Activation(act))
        m.add(Dropout(dropout))
    m.add(Dense(2))
    m.add(Activation('sigmoid'))
    opt = Adam(lr=lr, decay=decay)
    m.compile(loss='binary_crossentropy', optimizer=opt, metrics=my_metrics)
    return m

#def pick_best_pdb(up2pdb):
#    for up in up2pdb:
        
    
def get_pfp_generator_model_simple(model_arch_file, model_weights_file):
    with open(model_arch_file) as archfile:
        model = model_from_json(archfile.read())
    model.load_weights(model_weights_file)
    model2 = Sequential()
    print( "Reading model from:",model_arch_file)
    print( model.summary() )
    
    #for m in model.layers[:-5]:
    for m in model.layers:
        model2.add(m)
    for i in range(len(model.layers)-1, 0,-1):
        print (model2[i])
    return model2

def get_pfp_generator_model(model_arch_file, model_weights_file, output_layer):
    with open(model_arch_file) as archfile:
        model = model_from_json(archfile.read())
    model.load_weights(model_weights_file)

    my_layer =None
    for layer in model.layers:
        print(layer.name)
        if layer.name == output_layer:
            my_layer = layer
            print("      found")
    if type(my_layer)==type(None):
        print("ERROR: could not find this layer to create a PFP generator:",output_layer)
    if len(my_layer.output_shape)==5:
        return Model(inputs=model.input, outputs=GlobalAveragePooling3D()(my_layer.output))
    elif len(my_layer.output_shape)==4:#output_layer in ["img_conv_bottleneck", "dropout_2"]:
        return Model(inputs=model.input, outputs=GlobalAveragePooling2D()(my_layer.output))
    elif len(my_layer.output_shape)==3:
        return Model(inputs=model.input, outputs=GlobalAveragePooling1D()(my_layer.output))
    else:
        return Model(inputs=model.input, outputs=my_layer.output)

def get_pfp_lookup(pfp_generator, up2pdb_map, pngfolder=os.path.join("chembl","png"), png_suffix="_dist.png"):
    protprints = {}
    counter=0
    for up in up2pdb_map:
        print_progress(counter,len(up2pdb_map), prefix="Produce protein fingerprints from images:"); counter += 1
        pdbcodes = up2pdb_map[up]
        found =False
        for alternative in pdbcodes:
            
            pngloc = os.path.join(pngfolder,"%s%s"%(alternative.lower(),png_suffix))
            if not os.path.exists(pngloc): continue
            if not found:
                img = imread(pngloc, flatten=True)
                img = imresize(img,(128,128))
                if K.image_data_format()=="channels_last":
                    x = np.array([img])[...,np.newaxis]
                else:
                    x = np.array([img])[np.newaxis,...]
             
                last = pfp_generator.predict(x, batch_size=1, verbose=0)
                #print last.shape
                protprints[up] = last[0]
                found = True
    return protprints

def add_protein_fingerprints(dataframe, column_label, dictionary, pfp_length=512):
    pprints = []
    fpcount=0
    for acc in dataframe[column_label]:
        print_progress(fpcount,dataframe.shape[0], prefix="Adding protein fingerprints to data frame")
        if not acc in dictionary:
            fp=[np.nan for _ in xrange(pfp_length)]
        else:
            
            fp = dictionary[acc]
        assert len(fp)==pfp_length
        pprints.append(np.array(fp))
        fpcount+=1
    
    protein_fingerprints = pd.DataFrame(data=pprints, columns=["pfp%i"%i for i in xrange(pfp_length)])
    return dataframe.join(protein_fingerprints)

def create_decoy_set(compound_fingerprints, protein_fingerprints):
    decoy_pairs = pd.DataFrame()
    
    
    
    return decoy_pairs
    

if __name__=='__main__':
        
    chembl_loc = "chembl/chembl20_1uM.csv"
    up2pdb = chembl_loc.replace(".csv", "_up2pdb.txt")
    
    
    pdbfolder = os.path.abspath(os.path.join(os.path.split(chembl_loc)[0],"redo_pdb"))
    pdbfixedfolder = os.path.abspath(os.path.join(os.path.split(chembl_loc)[0],"fixed_pdb"))
    pngfolder = os.path.abspath(os.path.join(os.path.split(chembl_loc)[0],"png"))
    fetch_all_pdbs = True
    download_pdbs = False
    calc_pqr = False
    convert2image = False
    
    #model_basename="cath2bprgb_cnn_lenet3_128x128x3_20798_bt32_ep200_f5x5_cathcol2-3-4-5_loss_0.02_model"
    #model_basename="cath2bpanm_cnn_lenet3_128x128x1_20798_bt32_ep200_f5x5_cathcol2-3-4-5_loss_0.03_model"
    #model_basename="cath2bpelectro_cnn_lenet3_128x128x1_20798_bt32_ep200_f5x5_cathcol2-3-4-5_loss_0.03_model"
    model_basename="cath2bpdist_cnn_lenet3_128x128x1_20798_bt32_ep201_f5x5_cathcol2-3-4-5_loss_0.02_model"
    pdb_type = "pqr"
    png_suffix ="_dist.png"
    
    
    model_arch_file = "results/%s_arch.json"%model_basename
    model_weights_file = "results/%s_weights.hdf5"%model_basename
    model_history = "results/%s_history.json"%model_basename
    
    with open(model_history) as hist:
        training_history = ast.literal_eval(hist.read())
    metric ="loss"
    print( training_history.keys() )
    plt.plot( training_history[metric], linestyle="-", color="black", label=metric )
    plt.plot( training_history["val_"+metric], linestyle=":", color="black", label="val_"+metric)
    
    #plt.plot( [ref_val for i in xrange(len(d[metric]))], linestyle="--", color="black", label=ref_val_label )
    plt.xlabel("epoch")
    plt.ylabel("metric:"+metric)
    plt.legend(loc="lower right")
    plt.title("%s"%(model_basename))
    
    plt.show()
    
    if not os.path.exists(pdbfolder):
        os.mkdir(pdbfolder)
    
    df = pd.read_csv(chembl_loc)
    
    del df["tid"]
    del df["pref_name"]
    del df["chembl_id"]
    del df["target_pref_name"]
    
    print( df.shape)
    print( DFcounts(df) )
    
    
    
    if not os.path.exists(up2pdb):
        uniprot_ids=df["target_accession"].unique()
        
        pdb_results = mapUniprot2PDB(uniprot_ids)
        print( type(pdb_results) )
        with open(up2pdb, 'w') as outfile:
            outfile.write(pdb_results)
    else:
        with open(up2pdb) as pdbmapping:
            pdb_results = pdbmapping.read()
    
    lines = pdb_results.strip().split("\n")
    print( len(lines),"lines" )
    up2pdb_map={}
    for l in lines:
        up, pdb = l.strip().split()
        if up in up2pdb_map:
            up2pdb_map[up].append(pdb)
        else:
            up2pdb_map[up] = [pdb]
    
    # determine which PDB files to get. Either ALL, or a subset for each uniprot id.
    pdblist = []
    if fetch_all_pdbs:
        pdblist = set()
        for up in up2pdb_map:
            pdbs = up2pdb_map[up]
            for p in pdbs:
                
                pdblist.add(p)
        pdblist = list(pdblist)
    else:
        for ui in xrange(len(up2pdb_map.keys())):
            up = up2pdb_map.keys()[ui]
            p = up2pdb_map[up][0:min(5,len(up2pdb_map[up]))]
            pdblist.append(p)
    print( len(pdblist) )
    
    if download_pdbs:
        pdblist = download_representative_pdbs(pdblist, pdbfolder) 
    
    domlist = [os.path.split(f)[-1][:-4] for f in glob.glob(os.path.join("chembl",pdbfolder,"*.pdb"))]
    print( domlist[:5] )
    if calc_pqr:
        print( "calc_pqr")
        pdb2image.pdb2pqr(domlist, os.path.join("chembl","fixed_pdb"), os.path.join("chembl","pqr"),exe="./pdb2pqr",show_output=False)
        
    if convert2image:
        print( "convert2image" )
        convert_pdb_to_image(domlist,"chembl", "pqr", "png", 512, overwrite_files=True, show_pairplot=False, show_distplots=False, show_matrices=False, show_protein=False)
     
    # get protein fingerprints

    #import tensorflow as tf
    #with tf.device('/gpu'):
    
    # load Keras model
    pfp_generator = get_pfp_generator_model(model_arch_file, model_weights_file)
    # generate fingerprints and and assign them to corresponding Uniprot ID for a PDB structure (up2pdb_map)
    protprints = get_pfp_lookup(pfp_generator, up2pdb_map, pngfolder=os.path.join("chembl","png"), png_suffix=png_suffix)
    print( "proteins with fingerprint:",len(protprints.keys()) )

    # plot protein distance matrix
    dmat_pfp = fingerprint_distance_matrix([protprints[acc] for acc in protprints])
    plt.imshow(dmat_pfp,cmap=cm.binary, aspect="equal")
    plt.title("protein distance"); plt.ylabel("targets"); plt.xlabel("targets"); plt.show()  

    dat = df.copy()
    dat = df.sample(frac=0.01).reset_index(drop=True)
    #dat = dat.sample(frac=1).reset_index(drop=True)
    
    holdout_size=int(0.1*dat.shape[0])
    
    holdout_set = dat.copy().iloc[:holdout_size,:]
    
    
    dat = dat.iloc[holdout_size:,:]
    
    
    print( "dat",dat.shape )
    print( DFcounts(dat) )
    
    
    
    
    dat = add_compound_fingerprints(dat, dat["smiles"].values , fptype=2, nBits=512)
    print( "dat+fp",dat.shape)
    
    #for decoy, make copy of dataframe without pfps
    decoy = dat.copy()
    
    
    dat = add_protein_fingerprints(dat, "target_accession", protprints, pfp_length=512)
    print( "dat+fp+pfp",dat.shape )
    
    if False:
        from MulticoreTSNE import MulticoreTSNE as TSNE

        tsne = TSNE(n_jobs=4,perplexity=100)
        pfp_emb = tsne.fit_transform(dat.filter(regex="pfp"))
        
        
        plt.scatter(pfp_emb[0], pfp_emb[1])
        plt.show()
        
        #cfp_emb = tsne.fit_transform(dat.filter(regex="f"))
        #plt.scatter(cfp_emb[0], cfp_emb[1])
        #plt.show()
    
    #randomize order of protein fingerprints from original dataframe
    decoy_protein_fingerprints = dat.filter(regex="pfp").sample(frac=1).reset_index(drop=True)
    
    decoy = decoy.join(decoy_protein_fingerprints)
    
    
    
    
    dat["true_target"] = [1 for _ in xrange(dat.shape[0])]
    print( "dat+fp+pfp+t",dat.shape)
    decoy["true_target"] = [0 for _ in xrange(decoy.shape[0])]
    
    
    
    df = pd.concat([dat,decoy],axis=0,ignore_index=True)
    print( "df",df.shape)
    
    print( df.describe())
    print( DFcounts(df) )
    df = df.dropna(axis=0,how='any')
    print( "df+dropna",df.shape )
    #from sklearn.utils import shuffle
    #df = shuffle(df)
    df = df.sample(frac=1).reset_index(drop=True)
    
    
    print( df.shape)
    
                 
    X = df.iloc[:,4:-1].values 
    print( X.shape    )   
    y = df.iloc[:,-1].values
    print( y.shape )
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    #RANDOM FOREST
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    auc_score = roc_auc_score(y_test, y_pred_proba.T[1])
    accuracy = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = 2*(pre*rec)/(pre+rec)
    
    print( "Random Forest: AUC: %.3f   ACC:%.3f   F1:%.3f" % (auc_score,accuracy,f1) )
    
    plot_confusion_matrix( confusion_matrix(y_test, y_pred) , ["true","false"])
    
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, n_jobs=5)
    print( "Random Forest: CV accuracy scores",scores )
    mean_cv_acc = np.mean(scores)
    std_cv_acc = np.std(scores)
    print( "Mean CV accuracy: %.3f +/- %.3f"%(mean_cv_acc, std_cv_acc) )
    
    ref_val=mean_cv_acc
    ref_val_label="RF"
    
    
    
    

    # DEEP NEURAL NETWORK
    
    arch=[200,150,100,50,25]
    act ="relu"
    dropout=0.1
    ep=200
    bs=128
    my_metrics=['accuracy']
    lr=0.001
    decay=0.0
    m = create_binary_class_DNN(X_train.shape[1], arch=arch, act=act, dropout=dropout, my_metrics=my_metrics, lr=lr, decay=decay)
    
    callbacks = []
            
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=1.0/2, patience=5, min_lr=0.0)
    callbacks.append( rlrop )
    
    ea = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=20, verbose=1, mode='auto')
    callbacks.append(ea)
    
    print( m.summary() )
    
    if False:
        Y = pd.get_dummies(y).values
        print( "X",X.shape)
        print( "Y",y.shape)
        cv_model = KerasClassifier(build_fn=m, epochs=ep, batch_size=bs, verbose=1, callbacks=callbacks)
        print( cv_model)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
        results = cross_val_score(cv_model, X, Y,cv=10)#, cv=kfold)
        print( results)
        print( "10xCV score: %.3f +/- %.3f"%(results.mean(), results.std() ) )
        
    else:
        y_train = pd.get_dummies(y_train).values
        y_test = pd.get_dummies(y_test).values
        history = m.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test,y_test), callbacks=callbacks,shuffle=True)
        # plot the training and validation metrics over time
        train_metric = "acc"
        test_metric = "val_"+train_metric
        d = history.history
        print( d.keys() )
        
        plt.plot( d[train_metric], linestyle="-", color="black", label=train_metric )
        plt.plot( d[test_metric], linestyle=":", color="black", label=test_metric)
        if ref_val != None: plt.plot( [ref_val for i in xrange(len(d[train_metric]))], linestyle="--", color="black", label=ref_val_label )
        plt.xlabel("epoch")
        plt.ylabel("metric:"+train_metric)
        plt.legend(loc="lower right")
        plt.title("DNN=%s;act=%s;dr=%.2f"%(str(arch),act,dropout))
        plt.show()
    
        # important if you want to re-use your trained model in the future!
        print( "saving model")
        m.save_weights('model_weights.hdf5')
        with open('model_architecture.json','w') as wf:
            wf.write(m.to_json())
    
    
    
    
    
    
    
    
    holdout_set = add_compound_fingerprints(holdout_set, holdout_set["smiles"].values , fptype=2, nBits=512)
    holdout_set = add_protein_fingerprints(holdout_set, "target_accession", protprints, pfp_length=512)
    holdout_set = holdout_set.dropna(axis=0,how='any')
    print( holdout_set.shape )
    #ho_fp = holdout_set.filter(regex="FP")
    #ho_pfp = holdout_set.filter(regex="pfp")
    
    X_ho = holdout_set.iloc[:,4:].values#ho_fp.join(ho_pfp).values
    #assert X_ho.shape[1]==1024
    print( X_ho.shape )
    #X_ho = np.concatenate((ho_fp.values, ho_pfp.values), axis=1)
    print ()
    prediction = m.predict(X_ho)
    print( "prediction:",prediction.shape)
    # all are true targets!
    binary=[]
    for i in xrange(prediction.shape[0]):
        pair=list(prediction[i,:])
        maxval = max(pair)
        binary.append([pair[0]==maxval, pair[1]==maxval])
    binary = np.array(binary)
    print( "binary prediction",binary.shape)
    obs_true = holdout_set.shape[0]
    pred_true = binary[:,1].sum()
    predicted_fraction = pred_true / float(obs_true)
    
    print( "Fraction correctly predicted compound-target pairs (accuracy):", predicted_fraction)
