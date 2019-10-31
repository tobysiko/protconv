# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:24:01 2018

@author: ts149092
"""

import matplotlib as mpl

mpl.use('Agg')

import pandas as pd
import numpy as np
from ProtConv2D.predictChembl import get_pfp_generator_model, add_protein_fingerprints, fingerprint_distance_matrix, create_binary_class_DNN, fetchRCSBPDB, fetchRedoPDB
import os, ast,sys
from ProtConv2D.pdb2image_v2 import pdb2pqr, PDBToImageConverter
import glob
#%matplotlib inline

import biovec


from keras import backend as K

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import roc_curve, roc_auc_score, auc, recall_score, accuracy_score, confusion_matrix, silhouette_score, precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import operator
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.misc import imread,imresize
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
from sklearn.externals import joblib
from ProtConv2D.MLhelpers import print_progress, add_compound_fingerprints, plot_confusion_matrix, plot_ROC, plot_hist

import hdbscan


from sklearn.manifold import TSNE

from multiprocessing import cpu_count
        #from sklearn.preprocessing import LabelEncoder
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))


#Plotting parameter defaults
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
sns.set(style="ticks")
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=False)
plt.rcParams['figure.figsize']=(16,16)

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.MolDb import FingerprintUtils


def smiles2MACCS(x):
    m = Chem.MolFromSmiles(str(x))
    if m==None: return np.nan
    else:
        return np.array(MACCSkeys.GenMACCSKeys(m))

def smiles2Morgan(x, radius=2, nBits=1024):
    m = Chem.MolFromSmiles(str(x))
    if m==None: return np.nan
    else:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits))

def get_pfps_for_images_simple(pfp_generator, chain_list, pngfolder="png", png_suffix="_rgb.png", dim=128):
    protprints = {}
    counter=0
    for chain in chain_list:
        print_progress(counter,len(chain_list), prefix="Produce protein fingerprints from images:"); counter += 1
        
        pngloc = os.path.join(pngfolder,"%s%s"%(chain,png_suffix))
        if not os.path.exists(pngloc): continue

        img = imread(pngloc, flatten=True)
        img = imresize(img,(dim,dim))
        if K.image_data_format()=="channels_last":
            x = np.array([img])[...,np.newaxis]
        else:
            x = np.array([img])[np.newaxis,...]

        last = pfp_generator.predict(x, batch_size=1, verbose=0)
        #print last.shape
        protprints[chain] = last[0]
    return protprints

def get_pfps_for_images(pfp_generator, chain_list, layer_name="fc_last",pngfolder="png", png_suffix="_rgb.png", img_dim=299, nchannels=3):
    protprints = {}
    counter=0
    for chain in chain_list:
        
        pngloc = os.path.join(pngfolder,"%s%s"%(chain,png_suffix))
        print_progress(counter,len(chain_list), prefix="Produce protein fingerprints from images:", suffix=pngloc); counter += 1
        
        if not os.path.exists(pngloc): 
            print( "Does not exist:",pngloc)
            continue
        
        if nchannels==3:
                            
            img = imread(pngloc, flatten=False, mode="RGB")
            orig_size = img.shape[1]
            if img_dim != None and img_dim != img.shape[1]:
                img = imresize(img, size=(img_dim, img_dim), mode="RGB")
            x = np.array(img)
            if K.image_data_format()=="channels_first":
                x = np.reshape(img,(3,img_dim,img_dim))
            x = np.array(img)
            if K.image_data_format()=="channels_first":
                x = np.reshape(np.array(img),(3,img_dim,img_dim))
        else:   
            img = imread(pngloc, flatten=True)
            orig_size = img.shape[1]
            if img_dim != None and img_dim != img.shape[1]:
                img = imresize(img, size=(img_dim,img_dim), mode="L")
            if K.image_data_format()=="channels_last":
                x = np.array(img)[...,np.newaxis]
            else:
                x = np.array(img)[np.newaxis,...]

        last = pfp_generator.predict(np.array(x)[np.newaxis,...], batch_size=1, verbose=0)
        #print "last:",last.shape
        #print last
        #print type(last)
        protprints[chain] =  last.flatten()
        
    return protprints
    

if __name__=="__main__":   
    """
    Run workflow that uses a pre-trained neural network to extract protein structural fingerprints from PDB codes associated with compound targets in ChEMBL activity assays.
    Goal is to train binary pedictor of low/high activity or regressor of p-chembl value.

    Mandatory command line arguments in order:
    1) filename of saved Keras model truncated before "..._model...", e.g. kcc_xception_pt1_299x299x3_20798_bt128_ep200_f3x3_fc512x512_cath2345_4G_split0.40_lo0.019_vlo2.906
    2) compound fingerprint type, either MACCS or Morgan-x-y where x is the radius and y is the length
    3) predictor type, one of RF (RandomForest), ET (ExtraTrees), GB (Gradient Boosting)
    4) cross-validation fold, e.g. 5
    """
    if os.path.exists("/nasuni"):
        hpc_folder = "nasuni"
    else:
        hpc_folder = "hpc"
    
    data_loc = "/%s/projects/ts149092/chembl/"%(hpc_folder)
    pdbfolder = os.path.join(data_loc,"pdb")
    pqrfolder = os.path.join(data_loc,"pqr")
    chainsfolder = os.path.join(data_loc,"chains")
    pngfolder = os.path.join(data_loc,"png") # store cleaned chains
    
    # collect PDB ids
    csv_filename = os.path.join(data_loc,"chembl_act_traintest_set.csv")
    fasta_filename = os.path.join(data_loc,"chembl_seqs.fasta")
    biovec_fasta = "/%s/projects/ts149092/cath/cath-domain-seqs.fa"%(hpc_folder)
    biovec_model = "/home-uk/ts149092/biovec/cath_protvec.model"
    
    model_path=sys.argv[1]
    model_basename=os.path.basename(model_path)
    model_path=os.path.dirname(model_path)#"kcc_xception_pt1_299x299x3_20798_bt128_ep200_f3x3_fc512x512_cath2345_4G_split0.40_lo0.019_vlo2.906"
    print (model_basename)
    pdb_type = "pqr"
    png_suffix ="_rgb.png"
    img_dim = int(  model_basename.split("_")[3].split("x")[0] )
    pfp_length = int(  model_basename.split("_")[8].split("x")[-1] ) 

    last_layer = "fc_last"
    
    save_model = False
    
    
    cfp_type = sys.argv[2]#"Morgan" # "MACCS"
    morgan_radius = 3
    morgan_nBits = 512
    
    if "Morgan" in cfp_type:
        smiles_to_fp_mapper = smiles2Morgan
        tmp = cfp_type.split("-")
        if len(tmp)==3:
            morgan_radius = int(tmp[1])
            morgan_nBits = int(tmp[2])

        
    elif cfp_type == "MACCS":
        smiles_to_fp_mapper = smiles2MACCS
    assert any( [x in  cfp_type  for x in ["Morgan", "MACCS"] ] )

    print("cfp_type", cfp_type)
    
    predictor_type = sys.argv[3]#"GB"
    assert predictor_type in ["RF", "ET", "GB"]

    print("predictor_type", predictor_type)

    df = pd.read_csv(csv_filename)
    
    n_fixed_clusters = int(sys.argv[4]) #= fold cross validation


    print( df.shape)
    print( df.columns.tolist())
    print( len(df["molregno"].unique()), "compounds")
    print( len(df["target_chembl_id"].unique()), "targets")
    
    pdb_chains = df["best_match"].unique().tolist()
    
    print( len(pdb_chains), "chains expected")
    
 
    # get fasta file of seqeuences per chain
    pv = biovec.models.load_protvec(biovec_model)

    seq_ids=[]
    seqs=[]
    protvecs_flat=[]
    with open(fasta_filename) as f:
        tmp1 = f.read().split(">")
        for chunk in tmp1:
            tmp2 = chunk.split()
            if len(tmp2)==2:
                filename = tmp2[0]
                pdb_chain_code = os.path.basename(filename).split(".")[0]
                sequence = tmp2[1]
                pvec = np.array(pv.to_vecs(sequence))
                #print(pdb_chain_code, sequence, pvec.shape)
                seq_ids.append(pdb_chain_code)
                seqs.append(sequence)
                protvecs_flat.append(pvec.flatten())
    print("%i sequences found"%(len(seqs)))
    seq_df=pd.DataFrame(protvecs_flat, index=seq_ids)
    print(seq_df.info())
    
    seq_df.columns = ["pv"+str(i) for i in seq_df.columns.values]

    
    # get protein fingerprints
    ppdf_filename = model_basename+"_ppdf.csv"
    
    if os.path.exists(ppdf_filename):
        print( "Load protein fingerprints")
        ppdf = pd.read_csv(ppdf_filename, index_col=0)
    else:
        print( "Generate protein fingerprints")

        # check that all PDB and chains files are there
        pdb_found = []
        chains_found = []
        
        
        for x in pdb_chains:
            p = x[:4].lower()+".pdb"
            
            if not os.path.exists(os.path.join(pdbfolder, p)): 
                fetchRedoPDB(x[:4].lower(), pdbfolder)
            if not os.path.exists(os.path.join(pdbfolder, p)): 
                fetchRCSBPDB(x[:4].lower(), pdbfolder)
            if not os.path.exists(os.path.join(pdbfolder, p)): 
                print( "Could not find "+p )
                
            else: 
                import prody
                pdb_found.append(x[:4].lower())
                c=x[5]
                chainout = os.path.join(chainsfolder, x+".pdb")
                if not os.path.exists(chainout):
                
                    chain = prody.parsePDB(os.path.join(pdbfolder,p), chain=c )
        
                    if chain == None: 
                        chain = prody.parsePDB(os.path.join(pdbfolder,p), chain="_" )
                    
                    if chain != None: 
                        chain = chain.select("protein and heavy and altloc A _")
                    else:
                        print( "Unable to get chain from ", os.path.join(pdbfolder, p) )
                        continue
                    
        
                    prody.writePDB(chainout, chain)
                chains_found.append(x)
                
        
        print( len(chains_found),"chains found" )
        
        # convert pdb-chains to pqr
        # only necessary for electrostatic calculations
        
        if False: 
        
            pdb2pqr(chains_found, chainsfolder, pqrfolder, exe="/home-us/ts149092/pdb2pqr", show_output=False,show_error=True, ff="amber")
            
        
        # convert pqr-chains to images
        if False:
            print( "Convert target pdb to image")
            
            import prody
            prody.confProDy(verbosity="critical") #(‘critical’, ‘debug’, ‘error’, ‘info’, ‘none’, or ‘warning’)
        
            basefolder = "./"
        
            # initialize object with all arguments
            converter = PDBToImageConverter(
                            basefolder, 
                            os.path.join(basefolder,pqrfolder) , 
                            os.path.join(basefolder,pngfolder), 
                            os.path.join(basefolder,"npy"), 
                            pdb_chains,
                            img_dim=None, 
                            overwrite_files=False,
                            selection="protein and heavy",
                            pdb_suffix=".pqr",
                            channels=["dist","electro","anm"],
        
                            maxnatoms=10000
                        )
            # convert pdb to image
        
            for pi in range(len(chains_found)):
                p = chains_found[pi]
        
                converter.id_list = [p]
                try:
                    #print converter.selection
                    converter.convert_pdbs_to_arrays(show_pairplot=False, 
                                                    show_distplots=False, 
                                                    show_matrices=False, 
                                                    show_protein=False)
                except Exception as e:
                    print( e )
            
        
        
        model_arch_file = os.path.join(model_path, "%s_model_arch.json"%model_basename)
        model_weights_file = os.path.join(model_path, "%s_model_weights.hdf5"%model_basename)
        model_history = os.path.join(model_path, "%s_model_history.json"%model_basename)
        
        with open(model_history) as hist:
            training_history = ast.literal_eval(hist.read())
        metric ="loss"
        print( training_history.keys() )
        plt.plot( training_history[metric], linestyle="-", color="black", label=metric )
        plt.plot( training_history["val_"+metric], linestyle=":", color="black", label="val_"+metric)
        
        #plt.plot( [ref_val for i in range(len(d[metric]))], linestyle="--", color="black", label=ref_val_label )
        plt.xlabel("epoch")
        plt.ylabel("metric:"+metric)
        plt.legend(loc="lower right")
        plt.title("%s"%(model_basename))
        
        plt.savefig("input_model.png")
        #with tf.device( "/device:GPU:0"): #"/cpu:0"
    
        # load Keras model
        pfp_generator = get_pfp_generator_model(model_arch_file, model_weights_file, last_layer)
        layer_names = [layer.name for layer in pfp_generator.layers]
        print( layer_names )
        
        assert last_layer in layer_names
        # generate fingerprints and and assign them to corresponding Uniprot ID for a PDB structure (up2pdb_map)
        protprints = get_pfps_for_images(pfp_generator, chains_found, layer_name=last_layer, pngfolder=pngfolder, png_suffix=png_suffix, img_dim=img_dim)
        print( "proteins with fingerprint:",len(protprints.keys()) )
        
        
        #cg = sns.clustermap(protprints)
        #plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        #plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)
    
        
        #print cg.dendrogram_col.linkage # linkage matrix for columns
        #cg.dendrogram_row.linkage # linkage matrix for rows
        
        ppdf = pd.DataFrame(protprints).T
        ppdf.columns = ["pfp"+str(i) for i in range(pfp_length)]
        ppdf.to_csv(ppdf_filename)
    
    if False:
        # plot protein distance matrix
        dmat_pfp = fingerprint_distance_matrix([protprints[acc] for acc in protprints])
        plt.imshow(dmat_pfp,cmap=cm.binary, aspect="equal")
        plt.title("protein distance"); plt.ylabel("targets"); plt.xlabel("targets"); plt.show()  
    
        dat = df.copy()
    
        print( dat.shape, "before" )
        dat = dat[ dat["best_match"].isin(protprints.keys())]
        print( dat.shape, "after removing targets with missing fingerprints" )
    
        dat = df.sample(frac=0.01).reset_index(drop=True)
        #dat = dat.sample(frac=1).reset_index(drop=True)
    
        holdout_size=int(0.1*dat.shape[0])
    
        holdout_set = dat.copy().iloc[:holdout_size,:]
    
    
        dat = dat.iloc[holdout_size:,:]
    
    
        print( "dat",dat.shape)
        print( dat.info())
    
        
        
        
    if "pfp" not in ppdf.columns[0]:
        ppdf.columns = ["pfp"+str(i) for i in range(len(ppdf.columns))]
    
    print( "ppdf done")
    
    label_cols = ["protein_familiy_level1", "protein_familiy_level2", "protein_familiy_level3", "protein_familiy_level4"]
    
    print( len(df["best_match"].unique()) )
    dd = df.loc[:,["best_match"]+label_cols]
    
    ppdfvis = ppdf.merge(dd, how="left", left_index=True, right_on="best_match")
    ppdfvis = ppdfvis.set_index("best_match")
    
    
    print( "ppdf for vis done")
    
    
    X_pfp = ppdfvis.iloc[:,:-len(label_cols)].values
    #n_labels = len(ppdfvis.iloc[:,-len(label_cols)].unique())
    c = ppdfvis.iloc[:,-len(label_cols):]
    c = c.set_index(ppdfvis.index)
    print( c.shape )
    
    
    
    
    
    n_cpus = cpu_count()
    print( n_cpus, "CPUs available")
    
    if False:
    
        tsne_filename = model_basename+"_tsne.csv"
        if not os.path.exists(tsne_filename):
            from MulticoreTSNE import MulticoreTSNE as mcTSNE
            print( "Running t-SNE on proteins...")#,tsne )
            
            tsne = mcTSNE(n_jobs=n_cpus)#, perplexity=10,n_iter=2000, angle=0.5)  
            #tsne = TSNE()
            
            pfp_emb = tsne.fit_transform(X_pfp)
            tsne_scat = pd.DataFrame(pfp_emb, columns=["x1","x2"])
            tsne_scat = tsne_scat.set_index(ppdfvis.index)
            print( tsne_scat.shape )
            
            
            print( "Saving ", tsne_filename)
            tsne_scat.to_csv(tsne_filename)
        else:
            print( "Reading TSNE from file:",tsne_filename)
            tsne_scat = pd.read_csv(tsne_filename, index_col="best_match")
        
        tsne_scat = tsne_scat.merge(c, how='left', left_index=True, right_index=True)
        print("t-sne scat columns:",tsne_scat.columns)
        
        if True:
            print( "plotting protein t-sne")
            tsne_scat = tsne_scat.dropna()
            #sns.factorplot(x='x1', y='x2', data=tsne_scat, hue=label_cols[0])
            sns.lmplot( x='x1', y='x2', data=tsne_scat, hue=label_cols[0], fit_reg=False )
            plt.savefig(model_basename+"_tsne_0.png")

            sns.lmplot('x1', 'x2', data=tsne_scat, hue=label_cols[1], fit_reg=False)
            plt.savefig(model_basename+"_tsne_1.png")

            sns.lmplot('x1', 'x2', data=tsne_scat, hue=label_cols[2], fit_reg=False)
            plt.savefig(model_basename+"_tsne_2.png")
            
            sns.lmplot('x1', 'x2', data=tsne_scat, hue=label_cols[3], fit_reg=False)
            plt.savefig(model_basename+"_tsne_3.png")
        
        
        print( "tsne_scat")
        
        
        #print tsne_scat.to_csv("labeled_"+tsne_filename)
        
        
    


    ##################################################
    # generate compound fingerprints



    cfp_df= df.loc[:,["molregno","canonical_smiles","best_match"]+label_cols]
    cfp_df = cfp_df.drop_duplicates(subset="molregno").set_index("molregno")
    print( "cfp_df.head()")
    
    
    
    if "Morgan"in cfp_type: cfp_filename = model_basename+"_Morgan_%i_%i_cfp.csv"%(morgan_radius,morgan_nBits)
    elif cfp_type=="MACCS":cfp_filename = model_basename+"_MACCS_cfp.csv"
    
    if not os.path.exists(cfp_filename):
        
    
        if "Morgan" in cfp_type:
            cfp = cfp_df["canonical_smiles"].apply(lambda x: pd.Series( smiles_to_fp_mapper(x,morgan_radius,morgan_nBits) ))#.sample(n=1000, axis=0)
            cfp_length = morgan_nBits
        elif cfp_type =="MACCS":
            cfp = cfp_df["canonical_smiles"].apply(lambda x: pd.Series( smiles_to_fp_mapper(x) ))#.sample(n=1000, axis=0)
            cfp_length = cfp.shape[1]
        print("%s CFP with length %i"%(cfp_type, cfp_length))
        cfp.columns = ["cfp"+str(i) for i in range(cfp_length)]
        c = cfp_df.iloc[:,-len(label_cols):]
        c.set_index(cfp_df.index)
        cfp.to_csv(cfp_filename)
        
    else:
        cfp = pd.read_csv(cfp_filename, index_col="molregno")
    print( "cfp.head()"   )
    
    
    if False:
        print( "Running t-sne on compounds...")
        from MulticoreTSNE import MulticoreTSNE as mcTSNE
        tsne =  mcTSNE(n_jobs=n_cpus)#, perplexity=10,n_iter=2000, angle=0.5)  
        cfp_emb = tsne.fit_transform(cfp.values)
        cfp_emb.shape
        tsne_scat = pd.DataFrame(cfp_emb,columns=["x1","x2"])
        tsne_scat = tsne_scat.set_index(cfp.index)
    
        tsne_scat = tsne_scat.merge(c, how='left', left_index=True, right_index=True)
        tsne_scat = tsne_scat.dropna()
    
        print( "plotting compound t-sne")
        sns.lmplot( x='x1', y='x2', data=tsne_scat, hue=label_cols[0], fit_reg=False )
        sns.lmplot('x1', 'x2', data=tsne_scat, hue=label_cols[1], fit_reg=False)
        sns.lmplot('x1', 'x2', data=tsne_scat, hue=label_cols[2], fit_reg=False)
        sns.lmplot('x1', 'x2', data=tsne_scat, hue=label_cols[3], fit_reg=False)
    
        plt.savefig(model_basename+"_tsne_scat.png")
    
    X_df = df.loc[:,["molregno","best_match","protein_familiy_level1","pchembl_value","pchembl_class"]]\
        .merge(cfp, how="left", left_on="molregno", right_index=True)\
        .merge(ppdf, how="left", left_on="best_match", right_index=True)
    print( X_df.head() )
          
    
    
   
    
    
    if False:
        
    
        print( "Find best protein fingerprint clustering:")
    
    
        krange = range(3,20)
    
        X_pfp_clu = X_df.set_index("best_match").filter(regex="pfp").dropna().drop_duplicates()
        print( X_pfp_clu.shape)
    
        k2s = {}
        k2c = {}
        for k in krange:
            clu = KMeans(n_clusters=k)
            clu.fit(X_pfp_clu)
            score = silhouette_score(X_pfp_clu, clu.labels_, metric="euclidean")
            print( k,score)
            k2s[k]=score
            k2c[k]=clu.labels_
    
        best_k = max(k2s.iteritems(), key=operator.itemgetter(1))[0]
        print( "best k = ",best_k)
        X_pfp_clu["p_clusters"] = k2c[best_k]
        print( X_pfp_clu.head())
    elif True:
        X_pfp_clu = X_df.set_index("best_match").filter(regex="pfp").dropna().drop_duplicates()
        print( X_pfp_clu.shape)
        clu = KMeans(n_clusters=n_fixed_clusters)
        clu.fit(X_pfp_clu)
        score = silhouette_score(X_pfp_clu, clu.labels_, metric="euclidean")
        print(score)
        X_pfp_clu["p_clusters"] = clu.labels_
        print( X_pfp_clu.head())
    else:
    
    
        clu = hdbscan.HDBSCAN(metric="euclidean")
        X_pfp_clu = X_df.set_index("best_match").filter(regex="pfp").dropna().drop_duplicates()
        print( X_pfp_clu.shape )
        clu.fit(X_pfp_clu)
        print( clu.labels_.max()+1, "clusters found")
        X_pfp_clu["p_clusters"] = clu.labels_
        print( X_pfp_clu.head() )
    
    if True:
        if False:
            print( "Find best compound fingerprint clustering:")
            krange = range(3,5)
        
            X_cfp_clu = X_df.set_index("molregno").filter(regex="cfp").dropna().drop_duplicates().astype(np.int8)
            print( X_cfp_clu.shape)
            #print X_cfp_clu.head()
        
            k2s = {}
            k2c = {}
            for k in krange:
                clu = KMeans(n_clusters=k, n_init = 20, n_jobs=n_cpus)
                clu.fit(X_cfp_clu)
                score = silhouette_score(X_cfp_clu, clu.labels_, metric="rogerstanimoto")
                print( k,score)
                k2s[k]=score
                k2c[k]=clu.labels_
        
            best_k = max(k2s.iteritems(), key=operator.itemgetter(1))[0]
            print( "best k = ",best_k )
        
            X_cfp_clu["c_clusters"] = k2c[best_k]
            print( X_cfp_clu.head()    ) 
        elif True:
            X_cfp_clu = X_df.set_index("molregno").filter(regex="cfp").dropna().drop_duplicates()
            print( X_cfp_clu.shape)
            clu = KMeans(n_clusters=n_fixed_clusters)
            clu.fit(X_cfp_clu)
            score = silhouette_score(X_cfp_clu, clu.labels_, metric="rogerstanimoto")
            print(score)
            X_cfp_clu["c_clusters"] = clu.labels_
            print( X_cfp_clu.head())
        else:
        
            clu = hdbscan.HDBSCAN(metric="rogerstanimoto")
            X_cfp_clu = X_df.set_index("molregno").filter(regex="cfp").dropna().drop_duplicates()
            print( X_cfp_clu.shape)
            clu.fit(X_cfp_clu)
            print( clu.labels_.max()+1, "clusters found")
            X_cfp_clu["c_clusters"] = clu.labels_
            print( X_cfp_clu.head()  )
    
    
    
    
    # set up data sets for classification / regression
    
    
    X_df = df.loc[:,["molregno","best_match","protein_familiy_level1","pchembl_value","pchembl_class"]]\
                .merge(cfp, how="left", left_on="molregno", right_index=True)\
                .merge(ppdf, how="left", left_on="best_match", right_index=True)
        
    print (X_df.shape)
    
    X_df = X_df.merge(X_pfp_clu.filter(regex="p_clusters"), how="left", left_on="best_match", right_index=True, validate="many_to_one" )
    print (X_df.shape)
    X_df = X_df.merge(X_cfp_clu.filter(regex="c_clusters"), how="left", left_on="molregno", right_index=True, validate="many_to_one" )
    print (X_df.shape)    
    X_df.dropna(inplace=True)
    
    print (X_df.shape)
    print( "X_df  columns", X_df.columns.values )

    Xc_cmpd_only = X_df[X_df["pchembl_class"]!=1].filter(regex=r"cfp[0-9]*")
    Xr_cmpd_only = X_df.filter(regex=r"cfp[0-9]*")
    print ("cmpd_only",Xc_cmpd_only.shape, Xr_cmpd_only.shape)
    #Xc_cmpd_only.to_csv("Xc_cmpd_only.csv")
    #Xr_cmpd_only.to_csv("Xr_cmpd_only.csv")
    

    pdb_onehot = pd.get_dummies(X_df["best_match"])
    print("pdb_onehot:",pdb_onehot.shape)

    Xc_cmpd_tarcat = X_df[X_df["pchembl_class"]!=1].filter(regex=r"cfp[0-9]*").merge(pdb_onehot, how="left", left_index=True, right_index=True, validate="one_to_one"  )
    Xr_cmpd_tarcat = X_df.filter(regex=r"cfp[0-9]*").merge(pdb_onehot, how="left", left_index=True, right_index=True, validate="one_to_one"  )
    print ("cmpd_tarcat",Xc_cmpd_tarcat.shape, Xr_cmpd_tarcat.shape)

    #Xc_cmpd_tarcat.to_csv("Xc_cmpd_tarcat.csv")
    #Xr_cmpd_tarcat.to_csv("Xr_cmpd_tarcat.csv")
    
    
    #print(seq_df)

    Xc_cmpd_protvec = X_df[X_df["pchembl_class"]!=1].merge(seq_df, how="left", left_on="best_match", right_index=True, validate="many_to_one"  ).filter(regex=r"(cfp|pv)[0-9]*")
    #print(Xc_cmpd_protvec.info())
    Xr_cmpd_protvec = X_df.merge(seq_df, how="left", left_on="best_match", right_index=True, validate="many_to_one"  ).filter(regex=r"(cfp|pv)[0-9]*")
    print ("cmpd_protvec",Xc_cmpd_protvec.shape, Xr_cmpd_protvec.shape)
    #print(Xc_cmpd_protvec)
    #print(Xr_cmpd_protvec)
    #Xc_cmpd_protvec.to_csv("Xc_cmpd_protvec.csv")
    #Xr_cmpd_protvec.to_csv("Xr_cmpd_protvec.csv")
    

    Xc_cmpd_pfp = X_df[X_df["pchembl_class"]!=1].filter(regex=r"(cfp|pfp)[0-9]*")
    Xr_cmpd_pfp = X_df.filter(regex=r"(cfp|pfp)[0-9]*")
    print ("cmpd_pfp",Xc_cmpd_pfp.shape, Xr_cmpd_pfp.shape)
    #Xc_cmpd_pfp.to_csv("Xc_cmpd_pfp.csv")
    #Xr_cmpd_pfp.to_csv("Xr_cmpd_pfp.csv")
    #Xc_cmpd_pfp_tarcat = X_df[X_df["pchembl_class"]!=1].iloc[:,5:-2].merge(pd.get_dummies(X_df["best_match"]),how="left", left_index=True, right_index=True  )
    #Xr_cmpd_pfp_tarcat = X_df.iloc[:,5:-2].merge(pd.get_dummies(X_df["best_match"]),how="left", left_index=True, right_index=True  )
    #print ("cmpd_pfp_tarcat",Xc_cmpd_pfp.shape, Xr_cmpd_pfp.shape)


    yc = X_df[X_df["pchembl_class"]!=1].loc[:,"pchembl_class"]
    yc[yc==2] = 1
    
    yr = X_df.loc[:,"pchembl_value"]

    print("y",yc.shape, yr.shape)
    

    
    groups_c_pfp = X_df[X_df["pchembl_class"]!=1]["p_clusters"]
    groups_r_pfp = X_df["p_clusters"]
    groups_c_cfp = X_df[X_df["pchembl_class"]!=1]["c_clusters"]
    groups_r_cfp = X_df["c_clusters"]
    #n_splits=len(groups.unique())
    n_splits=n_fixed_clusters
    
    kf = KFold(n_splits=n_splits, shuffle=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    #shs = ShuffleSplit(n_splits=n_splits)
    gkf = GroupKFold(n_splits=n_splits)

    
    #assert np.isnan(X).sum()==0 and np.isnan(y).sum()==0
    

    if predictor_type=="RF":
        clf = RandomForestClassifier(oob_score=True)
        parameters_clf = { "max_depth":(None,10,50),  "max_features":(None,'sqrt'), "n_estimators":(100, 200) }
        reg = RandomForestRegressor(oob_score=True)
        parameters_reg = { "max_depth":(None,10,50),  "max_features":(None,'sqrt'), "n_estimators":(100, 200)}
    elif predictor_type=="ET":
        clf = ExtraTreesClassifier()
        parameters_clf = {"n_estimators":(100), "max_features":(None), "max_depth":(14,24,48) }
        reg = ExtraTreesRegressor()
        parameters_reg = {"n_estimators":(100), "max_features":(None), "max_depth":(14,24,48) }
    elif predictor_type=="GB":
        clf = GradientBoostingClassifier()
        parameters_clf = {"n_estimators":(100), "max_features":(None,"sqrt","log2"), "learning_rate":(0.05, 0.1, 0.2), "max_depth":(14,24,48) }
        reg = GradientBoostingRegressor()
        parameters_reg = {"n_estimators":(100), "max_features":(None,"sqrt","log2"), "learning_rate":(0.05, 0.1, 0.2), "max_depth":(14,24,48) }
    else:
        clf=None
        reg=None
    
    cv_list = [
                #n_splits, 
               "kfold",
               "GroupKFold_cfp",
               "GroupKFold_pfp"
               ]
    
    X_dict = {
              #"Xc1_cmpd_only":Xc_cmpd_only,
             "Xc2_cmpd_tarcat":Xc_cmpd_tarcat
             #"Xc3_cmpd_protvec":Xc_cmpd_protvec
            , "Xc4_cmpd_pfp":Xc_cmpd_pfp

            #, "Xr1_cmpd_only":Xr_cmpd_only
            , "Xr2_cmpd_tarcat":Xr_cmpd_tarcat
            #, "Xr3_cmpd_protvec":Xr_cmpd_protvec
            , "Xr4_cmpd_pfp":Xr_cmpd_pfp
            }

    from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score, r2_score, mean_absolute_error, mutual_info_score, matthews_corrcoef, make_scorer

    clf_scorers={"f1":make_scorer(f1_score),
                    "auc":make_scorer(roc_auc_score),
                    "acc":make_scorer(accuracy_score)
                    #"mc":make_scorer(matthews_corrcoef)
                    }

    reg_scorers={"r2":make_scorer(r2_score),
                    "nmae":"neg_mean_absolute_error",
                    "eva" :"explained_variance"
                    #"mi":make_scorer(mutual_info_score),
                    
                    }

    clf_refit = "f1"
    reg_refit = "r2"

    
    #results_dict={ k:{} for k in X_dict}
    master_df_clf = pd.DataFrame()
    master_df_reg = pd.DataFrame()
    
    for X_name in sorted(X_dict.keys()):
        for cv in cv_list:
            predictor=""
            this_cv = None
            grp = None
            
            if X_name[:2]=="Xc":
                predictor = "CLF"
                if cv=="kfold":
                    grp = None
                    this_cv = skf
                elif cv=="GroupKFold_pfp":
                    grp = groups_c_pfp.values
                    this_cv = gkf
                elif cv=="GroupKFold_cfp":
                    grp = groups_c_cfp.values
                    this_cv = gkf
                else:
                    grp = None
                    this_cv = cv
                
                print(X_name, str(cv), this_cv, grp, X_dict[X_name].shape, yc.shape)
                print (clf_scorers, clf_refit)

                gs = GridSearchCV(clf, parameters_clf, cv=this_cv, scoring=clf_scorers, refit=clf_refit, n_jobs=n_cpus)

                gs.fit(X=X_dict[X_name], y=yc, groups=grp)
                
                
            elif X_name[:2]=="Xr":
                predictor = "REG"
                if cv=="kfold":
                    grp = None
                    this_cv = kf
                elif cv=="GroupKFold_pfp":
                    grp = groups_r_pfp.values
                    this_cv = gkf
                elif cv=="GroupKFold_cfp":
                    grp = groups_r_cfp.values
                    this_cv = gkf
                else:
                    grp = None
                    this_cv = cv

                print(X_name, str(cv), this_cv, grp, X_dict[X_name].shape, yr.shape)
                print (reg_scorers, reg_refit)

                gs = GridSearchCV(reg, parameters_reg, cv=this_cv, scoring=reg_scorers, refit=reg_refit, n_jobs=n_cpus)
                gs.fit(X=X_dict[X_name], y=yr, groups=grp)
                
            else:
                predictor = ""
            
            if predictor != "":
                #print (sorted(gs.cv_results_.keys()) )
                df = pd.DataFrame(gs.cv_results_)
                tmp_name = "GSCV%i__%s_%s_%s_%s_%s"%(n_fixed_clusters, model_basename, predictor_type, cfp_type, X_name, str(cv).split("(")[0])
                df.to_csv(tmp_name+".csv")

                if save_model: joblib.dump(gs, tmp_name+".joblib") 

                fi = gs.best_estimator_.feature_importances_ 
                
                assert len(fi)==X_dict[X_name].shape[1]

                c_len = len(list(X_dict[X_name].filter(regex="cfp").columns.values))

                c_features = list(X_dict[X_name].iloc[:,:c_len].columns.values)
                p_features = list(X_dict[X_name].iloc[:,c_len:].columns.values)
                
                print("c_features:",c_features, len(c_features))
                print("p_features:",p_features, len(p_features))
                print("feature importances:", X_name)
                print (len(fi[:c_len]), len(fi[c_len:]) )
                
                c_fi_mean = np.mean(fi[:c_len] )
                c_fi_max  = np.max(fi[:c_len] )

                if not "cmpd_only" in X_name:
                    p_fi_mean = np.mean(fi[c_len:] )
                    p_fi_max  = np.max(fi[c_len:] )
                else:
                    p_fi_mean = 0.0
                    p_fi_max  = 0.0
                
                print("mean feature importance (compound, protein, p/c):")
                print(c_fi_mean, p_fi_mean, p_fi_mean/c_fi_mean )
                print("max feature importance (compound, protein, p/c):")
                print(c_fi_max, p_fi_max, p_fi_max/c_fi_max )

                print("best estimator:",gs.best_estimator_, gs.best_score_, gs.best_params_, gs.best_index_)
                print("out-of-bag generalization score:", gs.best_estimator_.oob_score_)
                
                if predictor=="CLF":
                    df = df[df["rank_test_%s"%clf_refit]==1]
                else:
                    df = df[df["rank_test_%s"%reg_refit]==1]
                
                df["task"] = X_name
                df["predictor"] = str(gs.best_estimator_).split("(")[0]
                df["CV"] = str(cv).split("(")[0]
                df["CV_class"] = str(this_cv).split("(")[0]
                df["splits"] = n_fixed_clusters
                df["c_len"] = len(fi[:c_len]) 
                df["p_len"] = len(fi[c_len:])
                df["mean_feat_import_cmpd"] = c_fi_mean
                df["mean_feat_import_prot"] = p_fi_mean
                df["mean_feat_import_p/c"] = p_fi_mean/c_fi_mean
                df["max_feat_import_cmpd"] = c_fi_max
                df["max_feat_import_prot"] = p_fi_max
                df["max_feat_import_p/c"] = p_fi_max/c_fi_max
                df["oob_score"] = gs.best_estimator_.oob_score_
                if predictor=="CLF":
                    df["best_score_%s"%clf_refit] = gs.best_score_
                else:
                    df["best_score_%s"%reg_refit] = gs.best_score_
                #df["best_params"] = str(gs.best_params_)
                
                
                print(predictor)

                df.to_csv(tmp_name+"__best.csv")
                #if predictor=="CLF":
                #    print(master_df_clf.info())
                #    if master_df_clf.empty:
                #        master_df_clf = df
                #    else:
                #        print("appending df")
                #        pd.concat([master_df_clf, df], ignore_index=True, verify_integrity=True, sort=True)

                #else:
                #    print(master_df_reg.info())
                #    if master_df_reg.empty:
                #        master_df_reg = df
                #    else:
                #        print("appending df")
                #        pd.concat([master_df_reg, df], ignore_index=True, verify_integrity=True, sort=True)
    #tmp_name = "GSCV%i__%s_%s_%s"%(n_fixed_clusters, model_basename, predictor_type, cfp_type)
    #master_df_clf.to_csv(tmp_name+"_best_CLF.csv")
    #master_df_reg.to_csv(tmp_name+"_best_REG.csv")
    
    
    files = glob.glob("GSCV5__*RF*Xc*__best.csv")
    df = pd.DataFrame()
    for f in files:
        tmp = pd.read_csv(f)
        df = df.append(tmp)
    df.to_csv("GSCV5_CLF_summary.csv")

    files = glob.glob("GSCV5__*RF*Xr*__best.csv")
    df = pd.DataFrame()
    for f in files:
        tmp = pd.read_csv(f)
        df = df.append(tmp)
    df.to_csv("GSCV5_REG_summary.csv")



