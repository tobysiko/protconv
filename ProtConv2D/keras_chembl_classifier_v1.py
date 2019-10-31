# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:30:46 2017

@author: ts149092
"""

from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import ProtConv2D.MLhelpers as MLhelpers
from ProtConv2D.keras_cath_classifier_v3 import CATH_Classifier
import prody
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.activations import softmax, linear, relu, sigmoid
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201           
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ProgbarLogger, Callback,\
    ModelCheckpoint
from keras.engine.topology import get_source_inputs
from keras.initializers import lecun_uniform, glorot_normal, he_normal  # , lecun_normal
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Permute, Lambda, RepeatVector, AveragePooling2D, MaxPooling1D, MaxPooling2D, Bidirectional, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Conv1D, Conv2D, Conv3D, AveragePooling2D, AveragePooling1D, UpSampling1D, UpSampling2D, Reshape, merge, multiply, concatenate, LSTM, CuDNNLSTM,CuDNNGRU
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.merge import add, multiply, dot, subtract, average, concatenate, maximum
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json, clone_model
from keras.preprocessing import sequence
from keras.preprocessing.text import hashing_trick, one_hot, Tokenizer, text_to_word_sequence
from keras.regularizers import l1, l2
from keras.utils import layer_utils, np_utils, multi_gpu_model, plot_model, Sequence, to_categorical
from keras.utils.data_utils import get_file
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.losses import categorical_crossentropy

import cv2 
import time, random, h5py, collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score, auc, recall_score, accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import homogeneity_score
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit,GroupKFold
import subprocess, os, sys, glob, warnings, argparse, random, math
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

import biovec

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.MolDb import FingerprintUtils

class ChEMBL_Classifier(CATH_Classifier):
    
    
    def __init__(self, root_path, image_folder="png", img_dim=-1, batch_size=16, epochs=10, 
                 model="fullconv", data_labels_filename="cath-dataset-nonredundant-S40.txt", label_columns=["pchembl_class"], 
                 png_suffix="_rgb.png", nchannels=3, sample_size=None, selection_filename=None, idlength=7, 
                 kernel_shape="3,3", dim_ordering="channels_last", valsplit=0.33, save_resized_images=False, 
                 outdir="results", use_pretrained=False, early_stopping="none", tensorboard=False, tbdir='./tblog', optimizer="adam", 
                 verbose=True, batch_norm=True, batch_norm_order="LAB", act="relu", learning_rate=-1, img_size_bins=[], dropout=0.25, 
                 img_bin_batch_sizes=[], flipdia_img=False, inverse_img=False, model_arch_file=None, 
                 model_weights_file=None, fc1=0, fc2=128, keras_top=True, ngpus=1, train_verbose=True, reduce_lr=True,
                 generic_label="chembl", show_images=False, img_bin_clusters=3, min_img_size=-1, max_img_size=-1,
                 h5_backend="h5py", h5_input="", use_img_encoder=False, use_img_decoder=False, dc_dec_weight=1.0, dc_dec_act="relu", dc_decoder_loss="mean_squared_logarithmic_error",
                 use_seq_encoder=False, use_seq_decoder=False,seq_decoder_loss="mean_squared_logarithmic_error", seq_dec_weight=1.0, seq_code_layer="fc_first", 
                 seq_dec_act="relu", manual_seq_maxlength=None, seq_enc_arch="lstm",seq_dec_arch="cnn", lstm_enc_length=128,lstm_dec_length=128,
                 cath_loss_weight=1.0, checkpoints=10, CATH_Y_labels_interpretable=False, cath_domlength_weight=1.0, domlength_regression_loss="mean_squared_logarithmic_error",
                 use_biovec=False, biovec_model="/home-uk/ts149092/biovec/cath_protvec.model",
                 seq_from_pdb=False, pdb_folder="dompdb", biovec_length=128, biovec_flatten=True, biovec_ngram=3,
                 generate_protvec_from_seqdict=True, seq_encoding="index", seq_padding='post',
                 conv_filter_1d=3, conv_filter_2d=(3,3),conv_filter_3d=(3,3,3),
                 crops_per_image=0, crop_width=32, crop_height=32, pre_shuffle_indices=True,
                 no_classifier=False, fullconv_img_clf_length=0, fullconv_seq_clf_length=0, valsplit_seed=None, use_lstm_attention=False, use_embedding=False,
                 dataset='chembl', merge_type='concatenate', classifier='dense', input_fasta_file="chembl_seqs.fasta",
                 mol_fp_length=2048, morgan_radius=3, morgan_type='bit',
                 names={"prot_ID":"best_match", "mol_ID":"molregno", "inputs":["main_input", "seq_input", "mol_input"], "clf_targets":[ "pchembl_class" ], "reg_targets":[ "pchembl_value" ]}
                 ):

        CATH_Classifier.__init__(self, root_path, image_folder=image_folder, img_dim=img_dim, batch_size=batch_size, epochs=epochs, 
                 model=model, data_labels_filename=data_labels_filename, label_columns=label_columns, 
                 png_suffix=png_suffix, nchannels=nchannels, sample_size=sample_size, selection_filename=selection_filename, idlength=idlength, 
                 kernel_shape=kernel_shape, dim_ordering=dim_ordering, valsplit=valsplit, save_resized_images=save_resized_images, 
                 outdir=outdir, use_pretrained=use_pretrained, early_stopping=early_stopping, tensorboard=tensorboard, tbdir=tbdir, optimizer=optimizer, 
                 verbose=verbose, batch_norm=batch_norm, batch_norm_order=batch_norm_order, act=act, learning_rate=learning_rate, img_size_bins=img_size_bins, dropout=dropout, 
                 img_bin_batch_sizes=img_bin_batch_sizes, flipdia_img=flipdia_img, inverse_img=inverse_img, model_arch_file=model_arch_file, 
                 model_weights_file=model_weights_file, fc1=fc1, fc2=fc2, keras_top=keras_top, ngpus=ngpus, train_verbose=train_verbose, reduce_lr=reduce_lr,
                 generic_label=generic_label, show_images=show_images, img_bin_clusters=img_bin_clusters, min_img_size=min_img_size, max_img_size=max_img_size,
                 h5_backend=h5_backend, h5_input=h5_input, use_img_encoder=use_img_encoder, use_img_decoder=use_img_decoder, dc_dec_weight=dc_dec_weight, dc_dec_act=dc_dec_act, dc_decoder_loss=dc_decoder_loss,
                 use_seq_encoder=use_seq_encoder, use_seq_decoder=use_seq_decoder,seq_decoder_loss=seq_decoder_loss, seq_dec_weight=seq_dec_weight, seq_code_layer=seq_code_layer, 
                 seq_dec_act=seq_dec_act, manual_seq_maxlength=manual_seq_maxlength, seq_enc_arch=seq_enc_arch,seq_dec_arch=seq_dec_arch, lstm_enc_length=lstm_enc_length,lstm_dec_length=lstm_dec_length,
                 cath_loss_weight=cath_loss_weight, checkpoints=checkpoints, CATH_Y_labels_interpretable=CATH_Y_labels_interpretable, cath_domlength_weight=cath_domlength_weight, domlength_regression_loss=domlength_regression_loss,
                 use_biovec=use_biovec, biovec_model=biovec_model, biovec_ngram=biovec_ngram,
                 seq_from_pdb=seq_from_pdb, pdb_folder=pdb_folder, biovec_length=biovec_length, biovec_flatten=biovec_flatten,
                 generate_protvec_from_seqdict=generate_protvec_from_seqdict, seq_encoding=seq_encoding, seq_padding=seq_padding,
                 conv_filter_1d=conv_filter_1d, conv_filter_2d=conv_filter_2d,conv_filter_3d=conv_filter_3d,
                 crops_per_image=crops_per_image, crop_width=crop_width, crop_height=crop_height, 
                 no_classifier=no_classifier, fullconv_img_clf_length=fullconv_img_clf_length, fullconv_seq_clf_length=fullconv_seq_clf_length, valsplit_seed=valsplit_seed, use_lstm_attention=use_lstm_attention, use_embedding=use_embedding,
                 input_fasta_file=input_fasta_file,
                 dataset=dataset, merge_type=merge_type, classifier=classifier )
        self.target_df = None
        self.mol_df = None
        self.act_df = None
        self.seq_dict = collections.OrderedDict()
        self.mol_dict = collections.OrderedDict()
        self.img_dict = collections.OrderedDict()
        self.morgan_type = morgan_type; assert self.morgan_type in ["bit", "hashed", "int"], self.morgan_type
        self.mol_fp_length = mol_fp_length
        self.morgan_radius = morgan_radius
        self.names = names
        self.clf_targets = self.names["clf_targets"]
        self.reg_targets = self.names["reg_targets"]
        self.clf_classes = {}
        self.my_losses = {}#collections.OrderedDict()
        self.my_loss_weights = {}#collections.OrderedDict()
        self.my_metrics = {}#collections.OrderedDict()
        
        self.prot_ID = self.names["prot_ID"]
        self.mol_ID = self.names["mol_ID"]
        self.pre_shuffle_indices = pre_shuffle_indices
        self.k_histories=[]
        self.k_modlabels=[]
        self.target_lists={}
    

    def import_chembl_dataset(self, img_h5_file=None,target_file=None, cmpd_file=None, activity_file=None):
        if img_h5_file != None:
            hf = h5py.File(img_h5_file, "r")
            print ("h5py keys:", hf.keys())
            self.X = hf["img"][()]
            print("Images loaded:",self.X.shape)
            #self.Ys = hf["class_labels"][()]
            self.labels = [s.strip().strip("'") for s in str(hf["pdb_chain_codes"][()] ).strip("[").strip("]").split(",")]
            
            print ("image labels loaded:", len(self.labels))
            print (self.labels[0])
            self.img_dict = {}
            # for i in range(len(self.labels)):
            #     print(self.labels[i], self.X[i,].shape)
            #     assert self.X[i,].shape == (256,256,3)
            self.img_dict = {self.labels[i]:self.X[i,] for i in range(len(self.labels))}
            self.sequences = [s.strip().strip("'") for s in str(hf["seq"][()]).strip("[").strip("]").split(",")]
            print ("sequences loaded:", len(self.sequences))
            print (self.sequences[0])
            self.seq_dict = {self.labels[i]:self.sequences[i] for i in range(len(self.labels))}
            if self.use_biovec:
                formatted = self.biovec_encoding()
            else:
                formatted = self.sequence_encoding()
            self.formatted_seq_dict = {self.labels[i]:formatted[i] for i in range(len(self.labels))}
            #print(self.sequences)
            self.not_found = []

            hf.close()

        if activity_file!=None:
            self.act_df = pd.read_csv(activity_file)
            print(self.act_df.shape)
            self.act_df = self.act_df.dropna(axis=0, how='any', subset=[self.names["prot_ID"], self.names["mol_ID"]]+self.names["clf_targets"]+self.names["reg_targets"])
            if True:# only two pchembl clases?
                self.act_df = self.act_df[ self.act_df["pchembl_class"]!=1] 
                self.act_df["pchembl_class"][self.act_df["pchembl_class"]==2] = 1
            print(self.act_df.shape)
            print(self.act_df.info())
            for t in self.clf_targets:
                assert t in self.act_df.columns, t
                print(t, sorted(self.act_df[t].unique()))
                self.clf_classes[t] = sorted(self.act_df[t].unique())
        
        if target_file != None:
            self.target_df = pd.read_csv(target_file)
            assert len(self.target_df[self.prot_ID].unique() ) == self.target_df.shape[0]
            print(self.target_df.info())

        if cmpd_file != None:
            self.mol_df = pd.read_csv(cmpd_file)
            assert len(self.mol_df[self.mol_ID].unique() ) == self.mol_df.shape[0]
            print(self.mol_df.info())
            self.prepare_molecules()
        

        proteins = sorted(self.act_df[self.prot_ID].unique() )
        for t in proteins:
            if not t in self.img_dict:
                print("remove proteins without structure", t, self.act_df.shape)
                self.act_df = self.act_df[ self.act_df[self.prot_ID] != t ]
                print(self.act_df.shape)
        
        # save single-column y before one-hot encoding
        for y in self.names["clf_targets"]:
            self.target_lists[y] = self.act_df[y]

        # replace ys with one-hot encoding
        self.act_df = pd.get_dummies(self.act_df, columns=self.clf_targets )
        print(self.act_df.info())
        
    def read_images(self, files, resized_img_loc=""):
        if self.img_dim == None: self.save_resized_images = False
        
        if resized_img_loc == "":
            resized_img_loc = "%s%sx%s" % (self.image_folder, self.img_dim, str(self.img_dim))
        # resized_img_loc = os.path.join(self.root_path, resized_img_loc)
        if self.save_resized_images and not os.path.exists(resized_img_loc):
            os.mkdir(resized_img_loc)
                
        img_data = []
        labels = []
        orig_sizes = []
        
        
        found_count = 0
        not_found_list = []
        
        if files==[]:
            files = glob.glob(self.image_folder + "/" + "???????%s" % self.png_suffix)
        
        
        ioerror_list = []
        
        
        print ("Reading %i image files..." % (len(files)))
        MLhelpers.print_progress(0, len(files))
        for fi in range(len(files)):
            f = files[fi]
            code = os.path.basename(f)[:self.idlength]
            try:
                
                if self.nchannels == 3:
                    
                    img = cv2.imread(f, cv2.IMREAD_COLOR)
                    if type(img)==type(None):
                        print ("ERROR: could not read image for "+code)
                        continue
                    orig_size = img.shape[1]
                    if self.min_img_size != -1 and orig_size < self.min_img_size: continue
                    if self.max_img_size != -1 and orig_size > self.max_img_size: continue
                    # print (img.shape)
                    # img = image.load_img(f, target_size=(outdim, outdim), grayscale=False)
                    if self.img_dim != None and self.img_dim < img.shape[1]:
                        img = cv2.resize(img, (self.img_dim, self.img_dim))
                    elif self.img_dim != None and self.img_dim >= img.shape[1]:
                        img = self.zero_pad_image(img, self.img_dim)
                    x = np.array(img)
                    if K.image_data_format() == "channels_first":
                        x = np.reshape(img, (3, self.img_dim, self.img_dim))
                else:   
                    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    orig_size = img.shape[1]
                    if self.min_img_size != -1 and orig_size < self.min_img_size: continue
                    if self.max_img_size != -1 and orig_size > self.max_img_size: continue
                    # print (img.shape)
                    # img = image.load_img(f, target_size=(outdim, outdim), grayscale=True)
                    
                    if self.img_dim != None and self.img_dim < img.shape[1]:
                        img = cv2.resize(img, (self.img_dim, self.img_dim))
                    elif self.img_dim != None and self.img_dim >= img.shape[1]:
                        img = self.zero_pad_image(img, self.img_dim)
                # print (cath_code, orig_size )
                
                if self.nchannels == 1:
                    if K.image_data_format() == "channels_last":
                        x = np.array(img)[..., np.newaxis]
                    else:
                        x = np.array(img)[np.newaxis, ...]
                else:
                    x = np.array(img)
                    if K.image_data_format() == "channels_first":
                        x = np.reshape(np.array(img), (3, self.img_dim, self.img_dim))

                
                if self.save_resized_images:
                    # print (x.shape,os.path.basename(f))
                    if self.nchannels == 1: tmp = np.reshape(x, (self.img_dim, self.img_dim))
                    else: 
                        if K.image_data_format() == "channels_last":
                            tmp = np.reshape(x, (self.img_dim, self.img_dim, self.nchannels))
                        else:
                            tmp = x
                    # print(x.shape)
                    cv2.imwrite(os.path.join(resized_img_loc, os.path.basename(f))+".png", tmp)
                
                if self.inverse_img:
                    for i in range(self.nchannels):
                        if K.image_data_format() == "channels_last":
                            matrix = x[:, :, i]
                            x[:, :, i] = 256 + (-255 * (matrix - np.amin(matrix)) / np.amax(matrix))
                        else:
                            matrix = x[i, :, :]
                            x[i, :, :] = 256 + (-255 * (matrix - np.amin(matrix)) / np.amax(matrix))
                
                if self.flipdia_img:
                    # print ("transpose")
                    if K.image_data_format() == "channels_last":
                        x = x[::-1, :, :]
                    else:
                        x = x[:, ::-1, :]
                    # print (x.shape)


                if self.img_dim != None and self.img_size_bins == []:
                    if x.shape[1] != self.img_dim:
                        print ("Unexpected image size %s! Expected %s" % (x.shape, self.img_dim))
                        continue
                        # x=np.pad(x,(32,32,3),mode="constant", constant_values=0)
                    
                    assert x.shape[1] == self.img_dim
                
                    img_data.append(x)
                    labels.append(str(code))
                    orig_sizes.append(orig_size)
                elif self.img_dim == None and self.img_size_bins == []:

                    
                
                    img_data.append(x)
                    labels.append(str(code))
                    orig_sizes.append(orig_size)
                else:
                    # https://stackoverflow.com/questions/8561896/padding-an-image-for-use-in-wxpython
                    # print (self.img_size_bins)
                    new_dim = self.img_size_bins[-1]
                    old_dim = x.shape[0]
                    for size in self.img_size_bins:
                        if old_dim <= size:
                            new_dim = size
                            break
                    # print ("Old dim: %i, new dim: %i"%(old_dim, new_dim))
                    if old_dim <= new_dim:
                        bg = np.zeros((new_dim, new_dim, self.nchannels), dtype=x.dtype)
                        if K.image_data_format() == "channels_first":
                            bg[:, :old_dim, :old_dim] = x
                        else:
                            bg[:old_dim, :old_dim, :] = x
                        x = bg
                    else:
                        print ("Image larger than biggest bin!", old_dim)
                        continue
                    img_data_s[new_dim].append(x)
                    labels_s[new_dim].append(code)

                found_count += 1
            except IOError as ioe:
                ioerror_list.append(f)
                print(f, ioe)
            except ValueError as ve:
                print(f, ve)
                    
        
            MLhelpers.print_progress(fi, len(files))
        print ("Found %i image files." % len(img_data))
        
        with open("ioerrors.txt", "w") as iof:
            iof.write("\n".join(ioerror_list) + "\n")
        # print (np.array(img_data).reshape((len(img_data),32,32,3)))
        # print (len(img_data))
        
        
        self.X = np.array(img_data)
        
        self.labels = labels
        self.not_found = not_found_list

    def export_chembl_dataset(self, expath, name):
        saved_files = []
        
        hdf5_path = os.path.join(expath, "ChEMBL_%s_%sx%sx%s_n%i.hdf5"%(name, self.img_dim, self.img_dim, self.nchannels, len(self.labels) ) )
        print ("Writing %s"%hdf5_path)
        
        data_shape = list(self.X.shape)
        data_shape[0] = 0
        print( data_shape )

        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset('img', data=self.X)
        hdf5_file.create_dataset('pdb_chain_codes', data=np.string_(self.labels) )
        hdf5_file.create_dataset('seq', data=np.string_(self.sequences))
        hdf5_file.close()

        saved_files.append( hdf5_path )
        return saved_files
    
    def get_morgan_fingerprint(self, smiles, inchi=None, radius=2, nBits=2048):
        m = Chem.MolFromSmiles(smiles)
        if type(m) in [type(None), np.nan]:
            if type(inchi) != type(None):
                m = Chem.MolFromInchi(inchi)
        assert type(m)==Chem.rdchem.Mol, type(m)  
        
        if self.morgan_type=="bit":
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits, useChirality=False, useBondTypes=True, useFeatures=False) )
        elif self.morgan_type=="hashed":
            fp = np.array(AllChem.GetHashedMorganFingerprint(m, radius, nBits=nBits, useChirality=False, useBondTypes=True, useFeatures=False) )
        else:
            fp = np.array(AllChem.GetMorganFingerprint(m, radius, useChirality=False, useBondTypes=True, useFeatures=False, useCounts=True) )
        ##print(fp)
        return fp

    def prepare_molecules(self):
        if self.verbose: print ("generating molecular fingerprints...")
        for i in self.mol_df.index:
            smiles = self.mol_df.loc[i, "canonical_smiles"]
            inchi = self.mol_df.loc[i,"standard_inchi"]
            molregno = self.mol_df.loc[i, "molregno"]
            #print(smiles, inchi)
            if str(smiles)=='nan' and str(inchi)=='nan':
                if self.verbose: print(self.act_df.shape, "removing mol without structure:"+str(molregno))
                self.act_df = self.act_df[ self.act_df["molregno"]!=molregno]
                if self.verbose: print(self.act_df.shape)
                continue
            
            mol_fp = self.get_morgan_fingerprint(smiles, inchi, self.morgan_radius, self.mol_fp_length)
            self.mol_dict[molregno] = mol_fp
    
    def get_model(self):
        
        if self.model == "fullconv":
            my_model = self.get_chembl_model(protein_branch_2d=self.use_img_encoder or self.use_img_decoder, protein_branch_1d=self.use_seq_decoder or self.use_seq_encoder)
        else:
            # obtain VGG16 model with dense layers and no pretrained weights
            print(K.image_data_format())
            if K.image_data_format() == "channels_last":
                input_tensor = Input(shape=(self.img_dim, self.img_dim, self.nchannels))
            else:
                input_tensor = Input(shape=(self.nchannels, self.img_dim, self.img_dim))

            KERAS_MODEL = None
            if self.model == "vgg16":           
                KERAS_MODEL = VGG16
            elif self.model == "vgg19":
                KERAS_MODEL = VGG19
            elif self.model == "inception":
                KERAS_MODEL = InceptionV3
            elif self.model == "xception":
                KERAS_MODEL = Xception
            elif self.model == "resnet50":
                KERAS_MODEL = ResNet50
            elif self.model == "densenet121":
                KERAS_MODEL = DenseNet121
            elif self.model == "densenet169":
                KERAS_MODEL = DenseNet169
            elif self.model == "densenet201":
                KERAS_MODEL = DenseNet201
            elif self.model == "inception_resnetv2":
                KERAS_MODEL = InceptionResNetV2
            elif self.model == "mobilenet":
                KERAS_MODEL = MobileNet
            elif self.model == "nasnetlarge":
                KERAS_MODEL = NASNetLarge
            elif self.model == "nasnetmobile":
                KERAS_MODEL = NASNetMobile
            else:
                my_model = None
            
            my_model = KERAS_MODEL(include_top=self.keras_top, weights="imagenet"if self.use_pretrained else None , input_tensor=input_tensor)
            
            print ("Layers from KERAS model:")
            print ([layer.name for layer in my_model.layers])
            # remove last output layer (prediction)
            del my_model.layers[-1]
            
            # grab loose ends
            x = my_model.layers[-1].output
            
            # add fully connected layers, collect inputs
            x, added_inputs = self.add_fclayers_to_model(x)
            inputs = [my_model.input] + added_inputs
            
            # add output layers
            outputs = self.add_outputs_to_model(x)
            
            # create new model with the modified outputs
            my_model = Model(inputs=inputs, outputs=outputs)
        
        
        arch_string = ""
        if self.model=='fullconv':
            if self.use_img_encoder or self.use_img_decoder:
                arch_string += "_img"
                if self.use_img_encoder: arch_string+="E"
                if self.use_img_decoder: arch_string+="D%.2f"% (self.dc_dec_weight)
                if self.fc1==0 and self.fc2==0 and not self.no_classifier:
                    arch_string += "CvClf"+str(self.fullconv_img_clf_length)
            if self.use_seq_encoder or self.use_seq_decoder:
                arch_string += "_seq"
                if self.use_seq_encoder: 
                    if self.seq_enc_arch=='cnn':
                        arch_string+="cE"
                    elif self.seq_enc_arch=='lstm':
                        arch_string+="lE"
                if self.use_seq_decoder: 
                    if self.seq_enc_arch=='cnn':
                        arch_string+="cD%.2f"% (self.seq_dec_weight)
                    elif self.seq_enc_arch=='lstm':
                        arch_string+="lD%.2f"% (self.seq_dec_weight)
                    
                if self.fc1==0 and self.fc2==0 and not self.no_classifier:
                    arch_string += "CvClf"+str(self.fullconv_seq_clf_length)
            
        clf_reg_string = "clf"
        # if "cath01_class" in self.nb_classes: clf_reg_string += "C"
        # if "cath02_architecture" in self.nb_classes: clf_reg_string += "A"
        # if "cath03_topology" in self.nb_classes: clf_reg_string += "T"
        # if "cath04_homologous_superfamily" in self.nb_classes: clf_reg_string += "H"
        # if "sequence_length" in self.nb_classes: clf_reg_string += "L"

        
        self.modlabel = "%s_%s%s_%s_Dr%s_%s_pt%i_%sx%sx%i_%i_bt%i_ep%i_f%ix%i_fc%ix%i_%s_%iG" % (
                            self.generic_label, self.model,
                            arch_string,
                            self.batch_norm_order,
                            str(self.dropout),
                            self.act,
                            int(self.use_pretrained),
                            str(self.img_dim), str(self.img_dim), self.nchannels, self.X.shape[0] if type(self.X)!=list else len(self.X),
                            self.batch_size, self.epochs, self.kernel_shape[0], self.kernel_shape[1],
                            self.fc1, self.fc2, clf_reg_string, self.ngpus
                        )
        
        
        
            
        
        
        print ("Summary string for model:")
        print (self.modlabel)
        plot_model(my_model, to_file=os.path.join(self.outdir, self.modlabel+'_model-diagram.png'),show_shapes=True, show_layer_names=True, rankdir='TB')#'LR'
        return my_model
                    
    def mol_encoder(self, mol_input):
        x = self.dense_block(mol_input, 256, n_layers=2, act='default', init='he_normal', layer_order="LAB", names=["mol_dense0_0","mol_dense0_1"])
        x = Dropout(self.dropout, name="do0")(x)
        x = self.dense_block(mol_input, 128, n_layers=2, act='default', init='he_normal', layer_order="LAB", names=["mol_dense1_0","mol_dense1_1"])
        x = Dropout(self.dropout, name="do1")(x)
        return x
    
    def get_chembl_model(self, protein_branch_2d=True, protein_branch_1d=True, mol_branch=True):

        if not self.load_model:
            model = self.get_fullconv_model(branch_2d=protein_branch_2d, branch_1d=protein_branch_1d)
            print(model.summary())
            del model.layers[-1]
            
        else:
            print ("Loading model...")
            print (self.model_arch_file)
            print (self.model_weights_file)
            with open(self.model_arch_file) as arch:
                model = model_from_json(arch.read())
            model.load_weights(self.model_weights_file)
            self.modlabel = self.model_arch_file.replace("_model_arch.json","")
            del model.layers[-5:-1]
        
        mol_input = Input(shape=(self.mol_fp_length, ), dtype='float32', name='mol_input')

        x = model.layers[-1].output
        x = concatenate([x, self.mol_encoder(mol_input)], name="concat_prot_mol")
        x = self.dense_block(x, 128, n_layers=2, act='default', init='he_normal', layer_order="LAB", names=["dense_block1_0","dense_block1_1"])
        x = Dropout(self.dropout,name="do2")(x)        
        
        if  not type(model.input) in [list,tuple]:
            inputs = [model.input, mol_input]
        else:
            inputs = model.input + [mol_input]
        
        outputs = self.add_chembl_outputs_to_model(x, img_decoder=None, seq_decoder=None)
        
        my_model = Model(input=inputs, output=outputs)

        print (my_model.summary())
        return my_model
    
    def add_chembl_outputs_to_model(self, x, img_decoder=None, seq_decoder=None):
        # x is the incoming layer from an existing model
        
        # collect outputs
        outputs = []
        if type(x)!=type(None):
            # attach four new output layers instead of removed output layer
            for l in self.clf_targets:
                outputs.append(Dense(len(self.clf_classes[l]), activation="softmax", name=l)(x))
                self.my_losses[l] = 'categorical_crossentropy'
                self.my_loss_weights[l] = 1.0
                self.my_metrics[l] = 'accuracy'
            for l in self.reg_targets:
                outputs.append(Dense(1, activation="linear", name=l)(x))
                self.my_losses[l] = 'mean_absolute_error'
                self.my_loss_weights[l] = 1.0
            
        
        # optionally, add a 2D decoder
        if self.use_img_decoder:
            outputs.append(img_decoder)
            self.my_losses['img_decoder'] = self.dc_decoder_loss
            self.my_losses[l] = self.dc_decoder_loss
            self.my_loss_weights["img_decoder"] = self.dc_dec_weight
        
        if self.use_seq_decoder:
            outputs.append( seq_decoder  )
            self.my_losses['seq_decoder'] = self.seq_decoder_loss
            self.my_losses[l] = self.seq_decoder_loss
            self.my_loss_weights["seq_decoder"] = self.seq_dec_weight  

        return outputs
    
    def kfold_train_chembl(self, n_splits=5, shuffle=False, mode="StratifiedKFold", random_state=None, target_column="pchembl_class", group_column=None):
        
        if mode=="StratifiedKFold":
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif mode=="StratifiedShuffleSplit":
            splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=self.valsplit, random_state=random_state)
        elif mode=="GroupKFold":
            splitter = GroupKFold(n_splits=n_splits)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        print(self.act_df.info())

        groups = group_column
        if type(group_column) != type(None):
            groups = self.act_df[group_column]

        count=1
        for train_index, val_index in splitter.split(self.act_df.index, self.target_lists[target_column], groups=groups):
            df_train_indices = self.act_df.iloc[train_index,].index
            df_val_indices = self.act_df.iloc[val_index,].index

            self.train_chembl(generic_label="F%i_"%count, train_indices=df_train_indices, val_indices=df_val_indices)
            
            self.k_histories.append(self.history)
            self.k_modlabels.append(self.modlabel)
            count+=1
        
        for i in range(n_splits):
            h = self.k_histories[i]
            m = self.k_modlabels[i]
            print("Fold %i >>> loss: %d.4; val_loss: %d.4 --- %s"%(i+1, h.history["loss"][-1], h.history["val_loss"][-1],  m  ) )



    def train_chembl(self, generic_label="", train_indices=[], val_indices=[]):
        # https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
        if self.ngpus == 1:
            self.keras_model = self.get_model()
        else:
            import tensorflow as tf

            with tf.device("/cpu:0"):
                orig_model = self.get_model()
                print ("Multi-GPU run - model on CPU:")
                print (orig_model)
                print(orig_model.summary())
                
            self.keras_model = multi_gpu_model(orig_model, gpus=self.ngpus)
        
        print (self.keras_model.summary())
        #print (self.keras_model.get_config())
        
        opt = self.get_optimizer()

        print(self.my_losses)
        print(self.my_loss_weights)
        print(self.my_metrics)

        self.keras_model.compile(
                                 loss=self.my_losses,
                                 optimizer=opt,
                                 metrics=self.my_metrics,
                                 loss_weights=self.my_loss_weights
                                 )
 
        start = time.time()



        if train_indices==[] or val_indices==[]:
            all_indices = self.act_df.index.values
            if self.pre_shuffle_indices:
                random.shuffle(all_indices)

            train_indices = all_indices[:int(math.ceil(len(all_indices)*(1.0-self.valsplit) ))]#  [int(i) for i in range(self.X.shape[0]) if i%2==0 ] 
            #random.shuffle(train_indices)

            val_indices = all_indices[int(math.ceil(len(all_indices)*(1.0-self.valsplit) )) : ]#[int(i) for i in range(self.X.shape[0]) if i%2!=0 ]
            #random.shuffle( val_indices )
            
            print("train_indices:",len(train_indices))
            print("val_indices:",len(val_indices))
            

        for v in val_indices:
            assert not v in train_indices,v

        training_generator = ChEMBL_Data_Generator(
                                                indices=train_indices,
                                                names=self.names,
                                                act_df=self.act_df,
                                                img_dict=self.img_dict,
                                                seq_dict=self.formatted_seq_dict, 
                                                mol_dict=self.mol_dict,
                                                batch_size=self.batch_size,
                                                is_train=True,
                                                crops_per_image=self.crops_per_image, 
                                                crop_width=self.crop_width, 
                                                crop_height=self.crop_height
                                                )   
        validation_generator = ChEMBL_Data_Generator(
                                                indices=val_indices,
                                                names=self.names,
                                                act_df=self.act_df,
                                                img_dict=self.img_dict,
                                                seq_dict=self.formatted_seq_dict, 
                                                mol_dict=self.mol_dict,
                                                batch_size=self.batch_size,
                                                is_train=False,
                                                crops_per_image=self.crops_per_image, 
                                                crop_width=self.crop_width, 
                                                crop_height=self.crop_height
                                                )    
        debug_factor=1.0 # so I don't have to wait too much...
        steps_per_epoch = int(debug_factor*max(1,self.crops_per_image) * math.ceil(len(train_indices) / self.batch_size) )
        validation_steps = int(debug_factor*max(1,self.crops_per_image) * math.ceil(len(val_indices)/ self.batch_size) )
        print("steps_per_epoch:",steps_per_epoch)
        print("validation_steps:",validation_steps)
        print("crops_per_image:",self.crops_per_image)

        self.history = self.keras_model.fit_generator(
                generator=training_generator,
                steps_per_epoch= steps_per_epoch,
                epochs=self.epochs,
                verbose=self.train_verbose,
                callbacks=self.get_callbacks(orig_model=orig_model if self.ngpus>1 else self.keras_model),
                validation_data=validation_generator,
                validation_steps=validation_steps,
                max_queue_size=20,
                workers=16,
                use_multiprocessing=True,
                shuffle=False 
                )
        
        print (len(self.history.history["loss"]), "epochs trained")
        for k in self.history.history.keys():
            print ("%s:\t%.3f" % (k, self.history.history[k][-1]))
        
        
        end = time.time()
        elapsed = end - start
        print ("Elapsed training time (h): %s"%str( (elapsed/60.0) / 60.0 ) )

        self.modlabel += "_split%.2f_lo%.3f_vlo%.3f" % (self.valsplit, self.history.history["loss"][-1], self.history.history["val_loss"][-1])
        # print self.history.history
        self.modlabel = generic_label + self.modlabel

        self.model_arch_file = os.path.join(self.outdir, "%s_model_arch.json" % self.modlabel)
        self.model_weights_file = os.path.join(self.outdir, "%s_model_weights.hdf5" % self.modlabel)
        
        if self.ngpus == 1:
            with open(os.path.join(self.outdir, "%s_model_history.json" % self.modlabel), "w") as hist:
                hist.write(str(self.history.history))
            self.keras_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(self.keras_model.to_json())
        else:
            with open(os.path.join(self.outdir, "%s_model_history.json" % self.modlabel), "w") as hist:
                hist.write(str(self.history.history))
            orig_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(orig_model.to_json())
            
        return self.modlabel   
    
    
    def visualize_chembl_image_data(self, show_plots=False, use_pixels=False, use_pfp=True, pfp_layername="fc_last", ndims=2, perp=100
        , early_exaggeration=4.0 
        , learning_rate=100, angle=0.5
        , n_iter=1000, rseed=123, pfp_length=512, marker_size=1, n_jobs=1, save_path="", do_cluster_analysis=False, clustering_method="kmeans", scores_dict={}): 
        
        
        assert clustering_method in ["kmeans", "hdbscan"]

        files_written = []
        
        if use_pfp:
            from ProtConv2D.predictChembl import get_pfp_generator_model
            pfp_generator = get_pfp_generator_model(self.model_arch_file, self.model_weights_file, pfp_layername)
            

            if self.use_seq_encoder and self.use_img_encoder:
                X_pfp = pfp_generator.predict([self.X, np.array(self.seqdata)])
            elif self.use_seq_encoder:
                X_pfp = pfp_generator.predict(np.array(self.seqdata))
            else:
                X_pfp = pfp_generator.predict(self.X)
        
        
        method = "barnes_hut"; assert method in ["barnes_hut", "exact"]
        metric = 'cosine'; assert metric in ["braycurtis", "canberra", "chebyshev", "cityblock",
                                                          "correlation", "cosine", "dice", "euclidean", "hamming",
                                                          "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski",
                                                          "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener",
                                                          "sokalsneath", "sqeuclidean", "yule"]
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        names = list(self.nb_classes.keys())#["Class", "Architecture", "Topology", "Homology"]
        #
        #classes = [self.Y[0], self.Y[1], self.Y[2], self.Y[3]]
        #nb_classes = [len(pd.Series(c).unique()) for c in classes]
                      
        if n_jobs > 1:
            from MulticoreTSNE import MulticoreTSNE as mcTSNE

            tsne = mcTSNE(n_jobs=n_jobs, perplexity=perp, n_iter=n_iter, angle=angle)                 
        else:
            tsne = TSNE(n_components=ndims,
                    random_state=rseed,
                    perplexity=perp,
                    early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate,
                    n_iter=n_iter,
                    angle=angle,
                    method=method,
                    metric=metric,
                    verbose=2
                    
                   )
        
        #print (nb_classes)
        
        if use_pixels and self.use_img_encoder:
            if self.img_dim!=None: 
                print (self.X.shape)
                X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1] * self.X.shape[2] * self.X.shape[3]))
                print (X.shape)
                emb = tsne.fit_transform(X)
            else:
                emb = tsne.fit_transform(self.X)
            # emb = tsne.embedding_
            
            kld = tsne.kl_divergence_
            print ("t-SNE --- Kullback-Leibler divergence after optimization:", kld)
            le = LabelEncoder()
            if ndims == 2:
                x1, x2 = zip(*emb)
            elif ndims == 3:
                x1, x2, x3 = zip(*emb)
            count = 0
            fig = plt.figure(figsize=(20, 20))
            plt.suptitle("input: protein image pixels of dim %i\nt-SNE with perplexity %.3f" % (self.img_dim, perp))
            

            #https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
            

            clusters = []
            for lk in list(self.nb_classes.keys() )[:4]:
                l = self.Ydict[lk]
                if len(l) == 0:continue
                
                labels = le.fit_transform(l)
                
                if do_cluster_analysis:
                    if clustering_method=="hdbscan":
                        import hdbscan
                        clu = hdbscan.HDBSCAN(metric="euclidean",min_cluster_size=self.nb_classes[lk])
                    elif clustering_method=="kmeans":
                        clu = KMeans(n_clusters=self.nb_classes[lk])
                    clu.fit(X)
                    score = homogeneity_score(labels, clu.labels_)
                    scores_dict["pixels_%s_homogeneity_score"%names[count]] = score
                    print( names[count]," score:",score)
                    clusters.append(clu.labels_)
                else:
                    score = -1
                
                cmap = cm.get_cmap(name="gist_ncar")
                c = [cmap(float(i) / self.nb_classes[lk]) for i in labels]
    
                # plt.subplot(4,1,count+1)
                
                if ndims == 2:
                    ax = fig.add_subplot(2, 2, count + 1)
                    # ax = plt.axes()
                    cax = ax.scatter(x1, x2, s=marker_size , c=c, lw=.5)  # ,cmap=colmap)
                elif ndims == 3:
                    ax = fig.add_subplot(2, 2, count + 1, projection='3d')
                    cax = ax.scatter(x1, x2, x3, s=marker_size, c=c, lw=.5)
                    ax.set_zlabel('ax3')                
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_title("%s (%i classes)%s" % (names[count], self.nb_classes[lk], " score=%s"%str(score) if score>=0 else ""))
                # ax.set_title(title)
                # if savename!=None: plt.savefig(savename,dpi=200)
                # plt.colorbar(cax)
                
                # plt.show()
                # plt.clf()
                count += 1
            if show_plots: plt.show()
            plt.savefig(os.path.join(save_path, self.modlabel+"_dataviz_pixels.png"))
            files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pixels.png"))
            df = pd.DataFrame({"tsne1":x1, "tsne2":x2, "Class":self.Ydict["cath01_class"], "Architecture":self.Ydict["cath02_architecture"], "Topology":self.Ydict["cath03_topology"], "Hom. Superfam.":self.Ydict["cath04_homologous_superfamily"]} 
                         ,columns=["Class","Architecture","Topology","Homology","tsne1","tsne2"], index=self.labels)
            if "sequence_length" in self.label_columns:
                domlen = self.Ydict["sequence_length"]
                cax = plt.scatter(x1, x2, s=marker_size, c=domlen, lw=.5)  # ,cmap=colmap)
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.colorbar( cax=cax )
                plt.title("input: protein image pixels of dim %i\nt-SNE with perplexity %.3f\ndomain length" % (self.img_dim, perp))               
                plt.savefig(os.path.join(save_path, self.modlabel+"_dataviz_pixels_domain_length.png"))
                files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pixels_domain_length.png"))
            
            if len(clusters)!=0:
                df["C_clusters"] = clusters[0]
                df["A_clusters"] = clusters[1]
                df["T_clusters"] = clusters[2]
                df["H_clusters"] = clusters[3]
            df.to_csv(os.path.join(save_path, self.modlabel+"_dataviz_pixels.csv"), index_label="cath_id" )
            files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pixels.csv"))
            
                
        # MLhelpers.scatter_plot(x1,x2)
        if use_pfp:
            emb = tsne.fit_transform(X_pfp)
            # emb = tsne.embedding_
            
            kld = tsne.kl_divergence_
            print ("t-SNE --- Kullback-Leibler divergence after optimization:", kld)
            le = LabelEncoder()
            if ndims == 2:
                x1, x2 = zip(*emb)
            elif ndims == 3:
                x1, x2, x3 = zip(*emb)
            count = 0
            fig = plt.figure(figsize=(20, 20))
            plt.suptitle("input: protein fingerprints of length %i\nt-SNE with perplexity %.3f" % (self.fc2, perp))
            clusters=[]
            for lk in list(self.nb_classes.keys() )[:4]:
                l = self.Ydict[lk]
                if len(l) == 0:continue
                labels = le.fit_transform(l)
                
                if do_cluster_analysis:
                    if clustering_method=="hdbscan":
                        import hdbscan
                        clu = hdbscan.HDBSCAN(metric="euclidean",min_cluster_size=self.nb_classes[count])
                    elif clustering_method=="kmeans":
                        clu = KMeans(n_clusters=self.nb_classes[lk])
                    clu.fit(X_pfp)
                    score = homogeneity_score(labels, clu.labels_)
                    scores_dict["pfprints_%s_homogeneity_score"%names[count]] = score
                    print( names[count]," score:",score)
                    clusters.append(clu.labels_)
                else:
                    score = -1
                
                cmap = cm.get_cmap(name="gist_ncar")
                c = [cmap(float(i) / self.nb_classes[lk]) for i in labels]
                
                # plt.subplot(4,1,count+1)
    
                if ndims == 2:
                    ax = fig.add_subplot(2, 2, count + 1)
                    # ax = plt.axes()
                    cax = ax.scatter(x1, x2, s=marker_size, c=c, lw=.5)  # ,cmap=colmap)
                elif ndims == 3:
                    ax = fig.add_subplot(2, 2, count + 1, projection='3d')
                    cax = ax.scatter(x1, x2, x3, s=marker_size, c=c, lw=.5)
                    ax.set_zlabel('ax3')       
                
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                # ax.set_title(title)
                ax.set_title("%s (%i classes)%s" % (names[count], self.nb_classes[lk], " score=%s"%str(score) if score>=0 else ""))               
                # if savename!=None: plt.savefig(savename,dpi=200)
                # plt.colorbar(cax)
                
                # plt.show()
                # plt.clf()
                count += 1
            
            
            if show_plots: plt.show()
            plt.savefig(os.path.join(save_path, self.modlabel+"_dataviz_pfprints.png" ))
            files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pfprints.png" ))
            df = pd.DataFrame({"tsne1":x1, "tsne2":x2, "Class":self.Ydict["cath01_class"], "Architecture":self.Ydict["cath02_architecture"], "Topology":self.Ydict["cath03_topology"], "Hom. Superfam.":self.Ydict["cath04_homologous_superfamily"]} 
                         ,columns=["Class","Architecture","Topology","Homology","tsne1","tsne2"], index=self.labels)
            
            if "sequence_length" in self.label_columns:
                domlen = self.Ydict["sequence_length"]
                print("domain length info available.")
                #domlen = self.Ys[-2]
                plt.scatter(x1, x2, s=marker_size, c=domlen, lw=.5)  # ,cmap=colmap)
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.colorbar()
                plt.title("input: protein fingerprints of length %i\nt-SNE with perplexity %.3f\ndomain length" % (self.fc2, perp))               
                plt.savefig(os.path.join(save_path, self.modlabel+"_dataviz_pfprints_domain_length.png" ))
                files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pfprints_domain_length.png"  ))
            
            if len(clusters)!=0:
                df["C_clusters"] = clusters[0]
                df["A_clusters"] = clusters[1]
                df["T_clusters"] = clusters[2]
                df["H_clusters"] = clusters[3]
            df.to_csv(os.path.join(save_path, self.modlabel+"_dataviz_pfprints.csv" ), index_label="cath_id" )
            files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pfprints.csv" ))
        plt.close('all')
        return files_written

class ChEMBL_Data_Generator(Sequence):

    def __init__(self, indices, names, act_df, img_dict, seq_dict, mol_dict, batch_size, is_train, shuffle=True, crops_per_image=0, crop_width=32, crop_height=32):
        self.indices=indices
        self.names=names
        
        self.act_df = act_df[act_df.index.isin(self.indices)] #act_df.loc[self.indices]#.reindex(self.indices)
        print(self.act_df.shape)
        self.act_df = self.act_df.dropna(axis=0, how='any', subset=["molregno", "best_match"])
        print(self.act_df.shape)
        self.img_dict = img_dict
        self.seq_dict = seq_dict
        self.mol_dict = mol_dict

        self.batch_size = batch_size
        self.crops_per_image = crops_per_image
        self.effective_batch_size = self.batch_size if crops_per_image<=0 else self.batch_size/self.crops_per_image

        self.is_train = is_train

        print("batch size ",self.batch_size )
        print("n crops ",self.crops_per_image)
        print("eff bs ",self.effective_batch_size)
        self.shuffle = shuffle
        
        self.crop_width = crop_width
        self.crop_height = crop_height
        #self.on_epoch_end()

    def __len__(self):
        return np.ceil( len(self.indices) / float(self.effective_batch_size))
    
    #def on_epoch_end(self):
    #    'Updates indexes after each epoch'
    #    self.indexes = np.arange(len(self.list_IDs))
    #    if self.shuffle == True:
    #        np.random.shuffle(self.indexes)
    
    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]
    
    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')

    #         # Store class
    #         y[i] = self.labels[ID]

    #     return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __getitem__(self, idx):
        batches_x = {}
        #print("training?", self.is_train)
        #print(idx, idx * self.effective_batch_size, (idx + 1) * self.effective_batch_size )
        batch_act = self.act_df.iloc[idx * self.effective_batch_size:(idx + 1) * self.effective_batch_size,]
        #print("batch_act", batch_act.shape)
        for input in self.names["inputs"]:
            if input=="main_input":
                batches_x[input] = np.array([self.img_dict[v] for v in batch_act[self.names["prot_ID"]]]) #] batch_act["best_match"].map(self.img_dict)# np.ndarray([  self.img_dict[ s ] for s in batch_act["best_match"] ])
            elif input=="seq_input":
                batches_x[input] = np.array([self.seq_dict[v] for v in batch_act[self.names["prot_ID"]]])#batch_act["best_match"].map(self.seq_dict)
            elif input=="mol_input":
                batches_x[input] = np.array([self.mol_dict[v] for v in batch_act[self.names["mol_ID"]]])#batch_act["molregno"].map(self.mol_dict)
            #print(input, batches_x[input].shape)
        
        batches_y = {}
        for x in self.names["clf_targets"]:
            batches_y[x] = batch_act.filter(regex=x) # TODO: why is x a list ????
            #print(x, batches_y[x].shape)
        for x in self.names["reg_targets"]:
            batches_y[x] = batch_act[x].values # TODO: why is x a list ????
            #print(x, batches_y[x].shape)


        return ( batches_x, batches_y )

class TimeHistory(Callback):
    def __init__(self):
        super(TimeHistory, self).__init__()

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        time_diff = time.time() - self.epoch_time_start
        logs['epoch_duration'] = time_diff
        self.times.append(time_diff)

class ModelCheckpointMultiGPU(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        cpu_model: the original model before it is distributed to multiple GPUs
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.    
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, cpu_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpointMultiGPU, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        
        self.cpu_model = cpu_model
        
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.cpu_model.save_weights(filepath, overwrite=True)
                        else:
                            self.cpu_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.cpu_model.save_weights(filepath, overwrite=True)
                else:
                    self.cpu_model.save(filepath, overwrite=True)

def seq_from_onehot(x, chars_to_remove="-"):
    chars = '-ACDEFGHIKLMNPQRSTVWYX'
    a = x.argmax(axis=1)
    
    max_i_chars = [chars[i] for i in a if chars[i] not in chars_to_remove]
    
    return "".join(max_i_chars)
            
def round_figures(x, n):
        if x == 0: return None
        """Returns x rounded to n significant figures."""
        return round(x, int(n - math.ceil(math.log10(abs(x)))))

if __name__ == "__main__":
    
    # PARSE ARGUMENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="cath", help="")
    ap.add_argument("--domlistname", default="cath-domain-list.txt", help="")
    ap.add_argument("--model", default="lenet", choices=["none", "unet","lenet", "lenet2", "lenet3", "vgg16", "vgg19", "inception", "xception", "resnet50", "inception_resnetv2", "mobilenet", "densenet121", "densenet169", "densenet201"], help="none means do not perform classfication.")
    ap.add_argument("--pngpath", default="png2", help="")
    ap.add_argument("--dim", type=int, default=-1, help="")
    ap.add_argument("--bs", type=int, default=16, help="")
    ap.add_argument("--ep", type=int, default=10, help="")  
    ap.add_argument("--kernel_shape", default="3,3", help="")
    ap.add_argument("--label_columns", default="2,3,4,5", help="")
    ap.add_argument("--selection_filename", default="cath-dataset-nonredundant-S40.txt", help="'None' for no selection (all domains used).")
    ap.add_argument("--png_suffix", default="_rgb.png", help="")
    ap.add_argument("--nchannels", type=int, choices=[1, 3], default=3, help="")
    ap.add_argument("--samples", type=int, default=0, help="")
    ap.add_argument("--label", default="cath2rgb", help="")
    ap.add_argument("--tsne_images", action="store_true", help="")
    ap.add_argument("--dimord", choices=["channels_first", "channels_last"], default="channels_last", help="")
    ap.add_argument("--valsplit", type=float, default=0.33, help="")
    ap.add_argument("--save_resized", action="store_true", help="")
    ap.add_argument("--outdir", default="results", help="")
    ap.add_argument("--use_pretrained", action="store_true", help="")
    ap.add_argument("--early_stopping", default="none", help="")
    ap.add_argument("--tensorboard", action="store_true", help="")
    ap.add_argument("--opt", choices=["sgd", "rmsprop", "adam"], default="adam", help="")
    ap.add_argument("--less", action="store_true", help="")
    ap.add_argument("--noBN", action="store_true", help="")
    ap.add_argument("--act", choices=["relu", "elu", "selu", "tanh", "sigmoid"], default="relu")
    ap.add_argument("--classnames", action="store_true", help="Print all class names to screen.")
    ap.add_argument("--lr", type=np.float32, default=-1, help="Learning rate. -1 means default of specified optimizer.")
    ap.add_argument("--dropout", type=np.float32, default=.25, help="Dropout fraction")
    ap.add_argument("--img_bins", default="none", help="Example: 16,32,64,128,256,512,1024")
    ap.add_argument("--img_bins_bs", default="none", help="")
    ap.add_argument("--flipdia_img", action="store_true", help="")
    ap.add_argument("--inverse_img", action="store_true", help="")
    ap.add_argument("--model_arch_file", default=None, help="")
    ap.add_argument("--model_weights_file", default=None, help="")
    ap.add_argument("--show_images", action="store_true", help="")

    args = ap.parse_args()
    print("Settings:", args)  
    
    clf = CATH_Classifier(args.path,
                            image_folder=args.pngpath,
                            img_dim=args.dim,
                            batch_size=args.bs,
                            epochs=args.ep,
                            model=args.model,
                            data_labels_filename=args.domlistname,
                            label_columns=args.label_columns,
                            png_suffix=args.png_suffix,
                            nchannels=args.nchannels,
                            sample_size=args.samples,
                            selection_filename=args.selection_filename,
                            kernel_shape=args.kernel_shape,
                            dim_ordering=args.dimord,
                            valsplit=args.valsplit,
                            save_resized_images=args.save_resized,
                            outdir=args.outdir,
                            use_pretrained=args.use_pretrained,
                            early_stopping=args.early_stopping,
                            tensorboard=args.tensorboard,
                            optimizer=args.opt,
                            verbose=not args.less,
                            batch_norm=not args.noBN,
                            act=args.act,
                            learning_rate=args.lr,
                            dropout=args.dropout,
                            flipdia_img=args.flipdia_img,
                            inverse_img=args.inverse_img,
                            img_size_bins=[] if args.img_bins in ["none", "None", "0", "[]"] else list(map(int, args.img_bins.split(","))),
                            img_bin_batch_sizes=[] if args.img_bins_bs in ["none", "None", "0", "[]"] else list(map(int, args.img_bins_bs.split(","))),
                            model_arch_file=args.model_arch_file,
                            model_weights_file=args.model_weights_file,
                            show_images=args.show_images,
                            min_img_size=-1,
                            max_img_size=299
                            
                        )
    if args.tsne_images: clf.visualize_image_data()
    
    # MAIN 
    if args.model != "none":
        clf.train(generic_label=args.label, load_model=False, show_classnames=args.classnames)
        
        if clf.history != None:
            clf.plot_curves(metrics=["loss"])
            #clf.plot_curves(metrics="acc")
        
    if False:
        sub = subprocess.Popen("nvidia-smi", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print (sub.stdout.read())
        print (sub.stderr.read())
    
