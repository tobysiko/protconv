from keras_cath_classifier_v3 import CATH_Classifier, ModelCheckpointMultiGPU, seq_from_onehot
import os
import pandas as pd
from rdkit.Chem import MolFromMol2File, MolToSmiles, rdmolops,MolFromSmiles,MACCSkeys
import numpy as np


class BindingPredictor(CATH_Classifier):

    def __init__(self, root_path, image_folder="png", img_dim=-1, batch_size=16,
    epochs=10, model="lenet",
    data_labels_filename="PDBBIND_general_PL_metadata.csv",
    label_columns="neg_log_meas", png_suffix="_rgb.png", nchannels=3,
    sample_size=None, selection_filename=None, idlength=7, kernel_shape="3,3",
    dim_ordering="channels_last", valsplit=0.33, save_resized_images=False,
    outdir="results", use_pretrained=False, early_stopping="none",
    tensorboard=False, optimizer="adam", verbose=True, batch_norm=True,
    act="relu", learning_rate=-1, img_size_bins=[], dropout=0.25,
    img_bin_batch_sizes=[], flipdia_img=False, inverse_img=False,
    model_arch_file=None, model_weights_file=None, fc1=512, fc2=512,
    keras_top=True, ngpus=1, train_verbose=True, generic_label="kbp",
    show_images=False, img_bin_clusters=3, min_img_size=-1,
    max_img_size=-1, h5_backend="h5py", h5_input="", dc_decoder=False,
    dc_dec_weight=1.0, dc_dec_act="relu",seq_encoder=False, seq_decoder=False,
    seq_dec_weight=1.0, seq_code_layer="fc_first", seq_dec_act="relu",
    manual_seq_maxlength=None, seq_enc_arch="cnn", bind_loss_weight=1.0,
    checkpoints=10):
        super().__init__(root_path, image_folder, img_dim, batch_size, epochs,
        model, data_labels_filename, label_columns, png_suffix, nchannels,
        sample_size, selection_filename, idlength, kernel_shape, dim_ordering,
        valsplit, save_resized_images, outdir, use_pretrained, early_stopping,
        tensorboard, optimizer, verbose, batch_norm, act, learning_rate,
        img_size_bins, dropout, img_bin_batch_sizes, flipdia_img, inverse_img,
        model_arch_file, model_weights_file, fc1, fc2, keras_top, ngpus,
        train_verbose, generic_label, show_images, img_bin_clusters,
        min_img_size, max_img_size, h5_backend, h5_input, dc_decoder,
        dc_dec_weight, dc_dec_act, seq_encoder, seq_decoder, seq_dec_weight,
        seq_code_layer, seq_dec_act, manual_seq_maxlength, seq_enc_arch,
        bind_loss_weight, checkpoints)
        
        self.label_columns = label_columns.split(",")
        
    
    def get_pdbbind_data(self, column, index="PDB code" ,filename="PDBBIND_general_PL_metadata.csv", ):
        df = pd.read_csv(filename)
        df.set_index(df[index])
        print (df.shape)
        self.Ys = {"neg_log_meas":df["neg_log_meas"]}
        self.labels = df.index 
        def molfile2smiles(p):
            mol2_name = os.path.join(self.root_path, p, "%s_ligand.mol2"%(p) )
            
            mol = MolFromMol2File(mol2_name)
            smiles = MolToSmiles(mol)
            return smiles
        def smiles2MACCS(x):
            m = MolFromSmiles(unicode(x))
            if m==None: return np.nan
            else:
                return np.array(MACCSkeys.GenMACCSKeys(m))    
        df['smiles'] = df.index.apply(lambda x: molfile2smiles(x))
        df = df["smiles"].apply(lambda x: pd.Series( smiles2MACCS(x) ))#.sample(n=1000, axis=0)
        
        print (df.shape)

    def get_chembl_data(self, data_filename="chembl_act_traintest_set.csv", prot_id_col="best_match", cmpd_id_col="molregno", class_col="pchembl_class"):
        protein_id_list=[]
        prot_img_dict=[]
        prot_seq_dict=[]
        cmpd_dict=[]
        #TODO: build generator that combines inputs for each activity value
        pass

    def get_output_labels(self):
        my_outputs = []
        my_losses = {}
        my_loss_weights = {}
        my_metrics = {}
        # attach four new output layers instead of removed output layer
        if "neg_log_meas" in self.label_columns: 
            name = "neg_log_meas"
            my_outputs.append(name)
            my_losses[name] = 'mean_squared_error'
            my_loss_weights[name] = self.cath_loss_weight
            my_metrics[name] = 'accuracy'
        
        
        return my_outputs, my_losses, my_loss_weights, my_metrics
    def get_amats(self):
        for p in self.labels:
            mol2_name = "%s_ligand.mol2"%(p)
            
            mol = MolFromMol2File(mol2_name)
            #smiles = MolToSmiles(mol)
            amat = rdmolops.GetAdjacencyMatrix(mol)
            # get atom types along diagonal
            #bondidxs = [ ( b.GetBeginAtomIdx(),b.GetEndAtomIdx() ) for b in mol.GetBonds() ]
            
            
        
    
            
            
            
            
            