# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:30:46 2017

@author: ts149092
"""

from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
from . import MLhelpers
#from ProtConv2D import Conv_Seq_Dec_Softmax_Layer#, VarSizeImageDataGenerator
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
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ProgbarLogger, Callback, ModelCheckpoint, History, CSVLogger
from keras.engine.topology import get_source_inputs
from keras.initializers import lecun_uniform, glorot_normal, he_normal, lecun_normal
from keras.layers import Masking, Input, Dense, TimeDistributed,SpatialDropout1D, SpatialDropout2D, Dropout, Activation, Flatten, Permute, Lambda, RepeatVector, AveragePooling2D, MaxPooling1D, MaxPooling2D, Bidirectional, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Conv1D, Conv2D, Conv3D, AveragePooling2D, AveragePooling1D, UpSampling1D, UpSampling2D, Reshape, merge, multiply, concatenate, LSTM, GRU, CuDNNLSTM,CuDNNGRU
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
plt.rcParams['figure.figsize']=(20,20)
import seaborn as sns
sns.set()
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
import subprocess, os, sys, glob, warnings, argparse, random, math
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

import biovec


class CATH_Classifier():
    ALLOWED_MODELS = ["none", "unet","unet2", "fullconv","seq2img", "seq2img_gan", "seq2img_gan_gen","lenet", "lenet2", "lenet3", "vgg16", "vgg19", 
                      "inception", "xception", "resnet50", "mobilenet", "densenet121", 
                      "densenet169", "densenet201", "inception_resnetv2", "nasnetlarge", "nasnetmobile"]
    
    def __init__(self, root_path, image_folder="png", img_dim=-1, batch_size=16, epochs=10, 
                 model="lenet", data_labels_filename="cath-dataset-nonredundant-S40.txt", label_columns="2,3,4,5", 
                 png_suffix="_rgb.png", nchannels=3, sample_size=None, selection_filename=None, idlength=7, 
                 kernel_shape="3,3", dim_ordering="channels_last", valsplit=0.33, save_resized_images=False, 
                 outdir="results", use_pretrained=False, early_stopping="none", tensorboard=False, tbdir='./tblog', optimizer="adam", 
                 verbose=True, batch_norm=True, batch_norm_order="LAB", act="relu", learning_rate=-1, img_size_bins=[], dropout=0.25, 
                 img_bin_batch_sizes=[], flipdia_img=False, inverse_img=False, model_arch_file=None, 
                 model_weights_file=None, fc1=512, fc2=512, keras_top=True, ngpus=1, train_verbose=True, reduce_lr=True,
                 generic_label="kcc", show_images=False, img_bin_clusters=3, min_img_size=-1, max_img_size=-1,
                 h5_backend="h5py", h5_input="", use_img_encoder=False, use_img_decoder=False, dc_dec_weight=1.0, dc_dec_act="relu", dc_decoder_loss="mean_squared_logarithmic_error",
                 use_seq_encoder=False, use_seq_decoder=False,seq_decoder_loss="mean_squared_logarithmic_error", seq_dec_weight=1.0, seq_code_layer="fc_first", 
                 seq_dec_act="relu", manual_seq_maxlength=None, seq_enc_arch="cnn",seq_dec_arch="cnn", lstm_enc_length=128,lstm_dec_length=128,
                 cath_loss_weight=1.0, checkpoints=10, CATH_Y_labels_interpretable=False, cath_domlength_weight=1.0, domlength_regression_loss="mean_squared_logarithmic_error",
                 use_biovec=False, biovec_model="/home-uk/ts149092/biovec/cath_protvec.model",
                 seq_from_pdb=False, pdb_folder="dompdb", biovec_length=128, biovec_flatten=True, biovec_ngram=3,
                 generate_protvec_from_seqdict=True, seq_encoding="index", seq_padding='post',
                 conv_filter_1d=3, conv_filter_2d=(3,3),conv_filter_3d=(3,3,3),
                 crops_per_image=0, crop_width=32, crop_height=32,
                 no_classifier=False, fullconv_img_clf_length=512, fullconv_seq_clf_length=512, valsplit_seed=None, use_lstm_attention=False, use_embedding=False,
                 classifier='dense', input_fasta_file="cath_pdb_to_seq.fasta",
                 dataset='cath', merge_type='concatenate', conv_dropout = 0.2, pad_value = 0,
                 blank_img_frac=0.0, seq_index_extra_dim=False):
        self.root_path = root_path
        assert os.path.exists(self.root_path), self.root_path
        self.image_folder = os.path.join(self.root_path, image_folder)
        self.img_dim = img_dim
        if self.img_dim == 0: self.img_dim = None
        self.batch_size = batch_size
        assert self.batch_size >= 1
        self.epochs = epochs
        assert self.epochs >= 1
        self.model = model
        assert self.model in self.ALLOWED_MODELS, "Model must be one of: %s" % (str(self.ALLOWED_MODELS)) + self.model
        if self.img_dim == -1:  # choose default dimensions for given model
            if self.model in ["lenet", "lenet2", "lenet3"]:
                self.img_dim = 32
            elif self.model in ["vgg16", "vgg19", "resnet50", "mobilenet", "densenet121", "densenet169", "densenet201", "nasnetmobile"]:
                self.img_dim = 224
            elif self.model in ["inception", "xception", "inception_resnetv2"]:
                self.img_dim = 299
            elif self.model in ["nasnetlarge"]:
                self.img_dim = 331
            elif self.model in ["unet","unet2","fullconv"]:
                self.img_dim = 224
            else:
                self.img_dim = None
        elif self.img_dim == 0:
            self.img_dim = None
        self.data_labels_filename = os.path.join(self.root_path, data_labels_filename)
        #assert os.path.exists(self.data_labels_filename), self.data_labels_filename
        
        self.dataset = dataset
        assert self.dataset in ['cath', 'chembl']

        self.valsplit_seed = valsplit_seed
        self.crops_per_image = crops_per_image
        self.crop_width = crop_width
        self.crop_height= crop_height
        
        if type(label_columns) == str:
            self.label_columns = list(map(int, label_columns.split(",")) )
        else:
            self.label_columns = label_columns
        print (self.label_columns)
        #assert len(self.label_columns) >= 1
        self.png_suffix = png_suffix
        self.sample_size = sample_size
        if self.sample_size == 0: 
            self.sample_size = None
        self.nchannels = nchannels
        if selection_filename in ["None", "none", None]:
            self.selection_filename = None
        else:
            self.selection_filename = os.path.join(self.root_path, selection_filename)
        self.idlength = idlength
        #self.kernel_shape = list(map(int, tuple(kernel_shape.split(","))))
        self.img_bin_clusters = img_bin_clusters
        K.set_image_data_format(dim_ordering)
        print(K.image_data_format())
        print(K.image_dim_ordering())
        self.fc1 = fc1
        self.fc2 = fc2
        self.conv_filter_1d = conv_filter_1d
        self.conv_filter_2d = conv_filter_2d
        self.kernel_shape = self.conv_filter_2d
        self.conv_filter_3d = conv_filter_3d
        self.keras_top = keras_top
        self.valsplit = valsplit
        self.save_resized_images = save_resized_images
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        self.use_pretrained = use_pretrained
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr
        self.tensorboard = tensorboard
        self.tbdir = tbdir
        self.optimizer = optimizer
        self.history = None
        self.verbose = verbose
        self.batch_norm = batch_norm
        self.batch_norm_order = batch_norm_order
        self.act = act
        self.learning_rate = learning_rate
        self.flipdia_img = flipdia_img
        self.inverse_img = inverse_img
        self.img_size_bins = img_size_bins
        self.img_bin_batch_sizes = img_bin_batch_sizes
        if len(self.img_size_bins) != len(self.img_bin_batch_sizes):
            self.img_bin_batch_sizes = [self.batch_size for _ in self.img_size_bins]
        self.dropout = dropout
        self.img_size_bins = []
        self.model_arch_file = model_arch_file
        self.model_weights_file = model_weights_file
        self.modlabel = "default_" + model
        self.load_model = False
        if self.model_arch_file != None and self.model_weights_file != None:
            self.load_model = True
            self.modlabel = os.path.split(self.model_arch_file)[-1].rename("_model_arch.json","")
        self.X = None
        self.Ys = None

        self.Ydict = collections.OrderedDict()
        self.labels = None
        self.not_found = None
        
        self.dX = None
        self.dYs = None
        self.dlabels = None
        
        self.nb_classes = None
        self.dnb_classes = None
        self.Ymats = None
        self.dYmats = None
        self.sn = None
        
        self.seqdict = {}
        self.sequences = []
        self.seqdata = None
        self.seq_maxlength = -1
        self.manual_seq_maxlength = manual_seq_maxlength
        self.seq_enc_arch = seq_enc_arch
        assert seq_enc_arch in ["single_conv","cnn","lstm","gru", "bilstm", "trans", "none"]
        self.seq_dec_arch=seq_dec_arch
        assert seq_dec_arch in ["cnn","lstm"]
        self.seq_encoding=seq_encoding
        assert seq_encoding in ["index","onehot","physchem"]
        self.use_embedding = use_embedding
        self.seq_index_extra_dim = seq_index_extra_dim

        self.lstm_enc_length=lstm_enc_length
        self.lstm_dec_length=lstm_dec_length
        
        self.use_biovec = use_biovec
        self.biovec_model = biovec_model
        self.seq_from_pdb = seq_from_pdb
        self.pdb_folder = pdb_folder
        self.generate_protvec_from_seqdict = generate_protvec_from_seqdict
        self.biovec_length = biovec_length
        self.biovec_flatten = biovec_flatten
        self.biovec_ngram = biovec_ngram

        self.min_img_size = min_img_size
        self.max_img_size = max_img_size
        
        self.use_img_encoder = use_img_encoder
        self.use_img_decoder = use_img_decoder
        self.dc_dec_weight = dc_dec_weight
        self.dc_dec_act = dc_dec_act
        self.dc_decoder_loss = dc_decoder_loss
        
        self.use_seq_encoder = use_seq_encoder
        self.use_seq_decoder = use_seq_decoder and use_seq_encoder
        self.seq_dec_weight = seq_dec_weight
        self.seq_code_layer = seq_code_layer
        self.seq_dec_act = seq_dec_act
        self.seq_decoder_loss = seq_decoder_loss
        self.use_lstm_attention = use_lstm_attention
        self.seq_padding = seq_padding
        assert self.seq_padding in ['pre', 'post']
        
        self.blank_img_frac = blank_img_frac
        self.fullconv_img_clf_length = fullconv_img_clf_length
        self.fullconv_seq_clf_length = fullconv_seq_clf_length
        self.no_classifier = no_classifier
        self.classifier = classifier
        assert classifier in ["conv", "dense", "none", None]
        self.merge_type = merge_type
        self.conv_dropout = conv_dropout
        self.cath_loss_weight = cath_loss_weight
        self.cath_domlength_weight = cath_domlength_weight
        
        self.keras_model = None
        self.ngpus = ngpus
        self.show_images = show_images
        self.h5_input = h5_input
        self.h5_backend = h5_backend
        
        self.checkpoints = checkpoints
        # self.batch_size = self.batch_size*ngpus
        self.train_verbose = train_verbose
        self.generic_label = generic_label
        if len(img_size_bins) > 0: self.img_size_bins = img_size_bins
        self.CATH_Y_labels_interpretable = CATH_Y_labels_interpretable

        self.domlength_regression_loss = domlength_regression_loss

        self.input_fasta_file = input_fasta_file
        self.cath_column_labels = [
            "Column 1:  CATH domain name (seven characters)",
            "Column 2:  Class number",
            "Column 3:  Architecture number",
            "Column 4:  Topology number",
            "Column 5:  Homologous superfamily number",
            "Column 6:  S35 sequence cluster number",
            "Column 7:  S60 sequence cluster number",
            "Column 8:  S95 sequence cluster number",
            "Column 9:  S100 sequence cluster number",
            "Column 10: S100 sequence count number",
            "Column 11: Domain length",
            "Column 12: Structure resolution (Angstroms) (999.000 for NMR structures and 1000.000 for obsolete PDB entries)"
                ]
        self.pad_value = pad_value
        self.aa2phys_dict={ 'A':{'hp': 1.8,  'mw':89.094/100.0 ,  'charge': 0.0, 'N':0, 'S':0, 'O':0, 'ar':0} ,# 1
                            'C':{'hp': 2.5,  'mw':121.154/100.0,  'charge': 0.0, 'N':0, 'S':1, 'O':0, 'ar':0} ,# 2
                            'D':{'hp':-3.5,  'mw':133.104/100.0,  'charge':-1.0, 'N':0, 'S':0, 'O':2, 'ar':0} ,# 3
                            'E':{'hp':-3.5,  'mw':147.131/100.0,  'charge':-1.0, 'N':0, 'S':0, 'O':2, 'ar':0} ,# 4
                            'F':{'hp': 2.8,  'mw':165.192/100.0,  'charge': 0.0, 'N':0, 'S':0, 'O':0, 'ar':1} ,# 5
                            'G':{'hp':-0.4,  'mw':75.067/100.0 ,  'charge': 0.0, 'N':0, 'S':0, 'O':0, 'ar':0} ,# 6
                            'H':{'hp':-3.2,  'mw':155.156/100.0,  'charge': 0.1, 'N':2, 'S':0, 'O':0, 'ar':1} ,# 7
                            'I':{'hp': 4.5,  'mw':131.175/100.0,  'charge': 0.0, 'N':0, 'S':0, 'O':0, 'ar':0} ,# 8
                            'K':{'hp':-3.9,  'mw':146.189/100.0,  'charge': 1.0, 'N':1, 'S':0, 'O':0, 'ar':0} ,# 9
                            'L':{'hp': 3.8,  'mw':131.175/100.0,  'charge': 0.0, 'N':0, 'S':0, 'O':0, 'ar':0} ,#10
                            'M':{'hp': 1.9,  'mw':149.208/100.0,  'charge': 0.0, 'N':0, 'S':1, 'O':0, 'ar':0} ,#11
                            'N':{'hp':-3.5,  'mw':132.119/100.0,  'charge': 0.0, 'N':1, 'S':0, 'O':1, 'ar':0} ,#12
                            'P':{'hp':-1.6,  'mw':115.132/100.0,  'charge': 0.0, 'N':0, 'S':0, 'O':0, 'ar':1} ,#13
                            'Q':{'hp':-3.5,  'mw':146.146/100.0,  'charge': 0.0, 'N':1, 'S':0, 'O':1, 'ar':0} ,#14
                            'R':{'hp':-4.5,  'mw':174.203/100.0,  'charge': 1.0, 'N':3, 'S':0, 'O':0, 'ar':0} ,#15
                            'S':{'hp':-0.8,  'mw':105.093/100.0,  'charge': 0.0, 'N':0, 'S':0, 'O':1, 'ar':0} ,#16
                            'T':{'hp':-0.7,  'mw':119.119/100.0,  'charge': 0.0, 'N':0, 'S':0, 'O':1, 'ar':0} ,#17
                            'V':{'hp': 4.2,  'mw':117.148/100.0,  'charge': 0.0, 'N':0, 'S':0, 'O':0, 'ar':0} ,#18
                            'W':{'hp':-0.9,  'mw':204.228/100.0,  'charge': 0.0, 'N':1, 'S':0, 'O':0, 'ar':2} ,#19
                            'Y':{'hp':-1.3,  'mw':181.191/100.0,  'charge': 0.0, 'N':0, 'S':0, 'O':1, 'ar':1} ,#20
                            'X':{'hp': 0.0,  'mw':1.0,            'charge': 0.0, 'N':0, 'S':0, 'O':0, 'ar':0}
                        }
    
    def rescale_images(self):
        tmp = np.ndarray( (self.X.shape[0], self.img_dim, self.img_dim, self.nchannels) )
        print(tmp.shape)
        for i in range(self.X.shape[0]):
            
                img = self.X[i,]
                
                orig_size = img.shape[1]
                
                if self.img_dim != None and self.img_dim < orig_size:
                    img = cv2.resize(img, (self.img_dim, self.img_dim))
                elif self.img_dim != None and self.img_dim >= orig_size:
                    img = self.zero_pad_image(img, self.img_dim)
                x = np.array(img)
                #print(x.shape)
                if K.image_data_format() == "channels_first":
                    x = np.reshape(img, (3, self.img_dim, self.img_dim))
                tmp[i,] = x
        print(tmp.shape)
        self.X = tmp
    
    def import_cath_dataset(self, img_h5_file=None, metadata_file=None):
        if img_h5_file != None:
           
            hf = h5py.File(img_h5_file, "r")
            print ("h5py keys:", hf.keys())
            self.X = hf["img"][()]
            if self.img_dim != self.X.shape[1]:
                print ("Rescaling images from %i to %i"%(self.X.shape[1],self.img_dim))
                self.rescale_images()
            print("Images loaded:",self.X.shape)
            #self.Ys = hf["class_labels"][()]
            self.labels = hf["cath_codes"][()]
            print ("image labels loaded:", len(self.labels))
            self.not_found = []

            hf.close()
            if self.blank_img_frac >0.0:
                self.randomly_blank_images(self.blank_img_frac) 
        
        if metadata_file != None:
            cath_dtypes = {
                    "index":str ,
                    "domain_sequence":str ,
                    "cath01_class":str ,
                    "cath02_architecture":str ,
                    "cath03_topology":str ,
                    "cath04_homologous_superfamily":str ,
                    "cath05_S35_cluster":str ,
                    "cath06_S60_cluster":str ,
                    "cath07_S95_cluster":str ,
                    "cath08_S100_cluster":str ,
                    "cath09_S100_count":int ,
                    "sequence_length":int ,
                    "resolution": float
            }

            
            df = pd.read_csv(metadata_file, dtype=cath_dtypes)
            
            

            if type(self.X) != type(None) and type(self.labels) != type(None):
                df_tmp = pd.DataFrame({"index":self.labels}, dtype=str)
                df = df_tmp.merge(df, how='left',left_on='index', right_on='index')            
            
            print("Metadata loaded:\n", df.info())

            
            self.nb_classes = collections.OrderedDict()
            for c in df.columns:
                #print(c)
                self.Ydict[c] = df[c]

                if c in ['cath01_class', "cath02_architecture", "cath03_topology", "cath04_homologous_superfamily",
                        "cath05_S35_cluster", "cath06_S60_cluster", "cath07_S95_cluster","cath08_S100_cluster"]:
                    self.nb_classes[c] = len(df[c].unique())
                    #print(c, self.nb_classes[c], len(df[c].unique()))
                
                if c == "domain_sequence":

                    self.seqdict = collections.OrderedDict()
                    for i in df.index:
                        cath_code = df.loc[i, "index"]
                        seq = df.loc[i, c]
                        length = df.loc[i, "sequence_length"]
                        assert len(seq) == length, "%i vs %i"%(len(seq) , length)
                        self.seqdict[cath_code]=seq
                    self.sequences = self.Ydict[c]
                    self.prepare_sequence_data()
                    
            if img_h5_file == None:
                self.labels=df.index.values

    def export_cath_dataset(self, expath, name, export_images=True, export_metadata=True):    
        saved_files = []
        if export_metadata:
            df = pd.DataFrame()
            df["index"] = self.labels
            df["domain_sequence"] = self.sequences
            df["cath01_class"] = self.Ys[0]
            df["cath02_architecture"] = self.Ys[1]
            df["cath03_topology"] = self.Ys[2]
            df["cath04_homologous_superfamily"] = self.Ys[3]
            df["cath05_S35_cluster"] = self.Ys[4]
            df["cath06_S60_cluster"] = self.Ys[5]
            df["cath07_S95_cluster"] = self.Ys[6]
            df["cath08_S100_cluster"] = self.Ys[7]
            df["cath09_S100_count"] = self.Ys[8]
            df["sequence_length"] = self.Ys[9]
            df["resolution"] = self.Ys[10]

            print(df.info())
            fname = os.path.join(expath, "CATH_metadata.csv")
            df.to_csv(  fname, index=False )
            saved_files.append( fname )

        if export_images:
            hdf5_path = os.path.join(expath, "CATH_%s_%sx%sx%s_n%i.hdf5"%(name, self.img_dim, self.img_dim, self.nchannels, len(self.labels) ) )
            print ("Writing %s"%hdf5_path)
           
            data_shape = list(self.X.shape)
            data_shape[0] = 0
            print( data_shape )

            hdf5_file = h5py.File(hdf5_path, mode='w')
            hdf5_file.create_dataset('img', data=self.X)
            hdf5_file.create_dataset('cath_codes', data=self.labels)
            hdf5_file.close()

            saved_files.append( hdf5_path )
        return saved_files

    def get_class_dict(self, name="cath-names.txt"):
        """
        Create dictionary of CATH hierarchy ID (numbers separaterd by dots) to the full name at the lowest level (where spaces in the name will be replaced by underscores).
        Example: the line 
        3.15    1ewfA01    :Super Roll
        will be converted to
        "3.15": {"domain":"1ewfA01", "name": ":Super_Roll"

        """
        d = {}
        with open(os.path.join(self.root_path, name)) as txt:
            lines = txt.readlines()
            for l in lines:
                if l[0] != "#":
                    tmp = l.split()
                    if len(tmp) >= 3:
                        d[tmp[0]] = {"domain":tmp[1], "name":"_".join(tmp[2:])}
        return d

    # def categorical_crossentropy_axis0(self, y_true, y_pred):
    #     return tf.keras.backend.categorical_crossentropy(y_true, y_pred, axis=0)
    # def categorical_crossentropy_axis1(self, y_true, y_pred):
    #     return tf.keras.backend.categorical_crossentropy(y_true, y_pred, axis=1)
    # def categorical_crossentropy_axis2(self, y_true, y_pred):
    #     return tf.keras.backend.categorical_crossentropy(y_true, y_pred, axis=2)
    
    def getTrainTestSetMulti_h5(self):
        
        if self.h5_backend == "h5py":
            import h5py
            if self.img_dim != None and self.img_size_bins == []:
                hf = h5py.File(self.h5_input, "r")
                print ("h5py keys:", hf.keys())
                self.X = hf["img"][()]
                self.Ys = hf["class_labels"][()]
                self.labels = hf["cath_codes"][()]
                self.not_found = []
                hf.close()
            else:
                h5_input_list = self.h5_input.split(",")
                assert len(h5_input_list) == len(self.img_size_bins), "%i vs %i" % (len(h5_input_list), len(self.img_size_bins))
                self.dX = {}
                self.dYs = {}
                self.dlabels = {}
                for si in range(len(self.img_size_bins)):
                    s = self.img_size_bins[si]
                    print(si, s, h5_input_list[si])
                    hf = h5py.File(h5_input_list[si], "r")
                    self.dX[s] = hf["img"][()]
                    self.dYs[s] = np.array(hf["class_labels"][()])
                    self.dlabels[s] = hf["cath_codes"][()]
                    self.not_found = []
                    
                    hf.close()
        else:  # pytables
            import tables
            if self.img_dim != None and self.img_size_bins == []:
                hf = tables.open_file(self.h5_input, mode='r')
                self.X = hf.root.img
                self.Ys = hf.root.class_labels
                self.labels = hf.root.cath_codes
                self.not_found = []
                hf.close()
            else:
                h5_input_list = self.h5_input.split(",")
                assert len(h5_input_list) == len(self.img_size_bins), "%i vs %i" % (len(h5_input_list), len(self.img_size_bins))
                self.dX = {}
                self.dYs = {}
                self.dlabels = {}
                self.not_found = []
                for si in range(len(self.img_size_bins)):
                    s = self.img_size_bins[si]
                    hf = tables.open_file(h5_input_list[si], mode='r')
                    self.dX[s] = hf.root.img
                    self.dYs[s] = np.array(hf.root.class_labels)
                    self.dlabels[s] = hf.root.cath_codes
                    
                    hf.close()
        
    def getTrainTestSetMulti(self, resized_img_loc="", files=[], propose_image_bins=False):
        if self.img_dim == None: self.save_resized_images = False
        
        if resized_img_loc == "":
            resized_img_loc = "%s%sx%s" % (self.image_folder, self.img_dim, str(self.img_dim))
        # resized_img_loc = os.path.join(self.root_path, resized_img_loc)
        if self.save_resized_images and not os.path.exists(resized_img_loc):
            os.mkdir(resized_img_loc)
        
        mydoms = set()
        if self.selection_filename != None:
            
            with open(self.selection_filename) as mydomfile:
                tmp = mydomfile.readlines()
                for t in tmp:
                    if len(t.strip()) == self.idlength:
                        mydoms.add(t.strip())
            print (len(mydoms), "doms selected from file")
        
        # file is : cath-domain-list.txt
        # 12 columns should be present, breaking down cath hierarchy as in self.cath_column_labels
        dom_dict = {}
        print ("Reading:", self.data_labels_filename)
        with open(self.data_labels_filename) as f:
            for l in f:
                if self.data_labels_filename in [os.path.join(self.root_path, x) for x in ["cath-domain-list.txt"]]:
                    line = l.decode("utf-8-sig")
                    #print (line)
                    if line[0] != '#':
                        tmp = [s.strip() for s in line.strip().split() ]

                        if self.data_labels_filename in [os.path.join(self.root_path, x) for x in ["cath-domain-list.txt"]]:
                            assert len(tmp) == 12, str(tmp)
                        if self.selection_filename != None and not tmp[0] in mydoms:
                            continue
                        else:
                            dom_dict[tmp[0]] = tmp[1:]
                else:
                    line = l
                    tmp = line.strip()
                    dom_dict[tmp]=[]


                
                
        img_data = []
        classes = []
        labels = []
        orig_sizes = []
        img_data_s = {s:[] for s in self.img_size_bins}
        classes_s = {s:[] for s in self.img_size_bins}
        labels_s = {s:[] for s in self.img_size_bins}
        
        found_count = 0
        not_found_list = []
        
        if files==[]:
            files = glob.glob(self.image_folder + "/" + "???????%s" % self.png_suffix)
        
        #class_dict = self.get_class_dict()# get the intepretable names of domains ### TODO does not return correct domains!!
        
        ioerror_list = []
        
        if self.sample_size != None:
            files = np.random.choice(files, self.sample_size)
        print ("Reading %i image files..." % (len(files)))
        MLhelpers.print_progress(0, len(files))
        for fi in range(len(files)):
            f = files[fi]
            cath_code = os.path.basename(f)[:self.idlength]
            if cath_code in dom_dict:
                try:
                    
                    if self.nchannels == 3:
                        
                        img = cv2.imread(f, cv2.IMREAD_COLOR)
                        if type(img)==type(None):
                            print ("ERROR: could not read image for "+cath_code)
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
                    # x = image.img_to_array(img)
                    # print (x.shape)
                    # x = x.T
                    
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
                    

                    #if self.CATH_Y_labels_interpretable:
                    #    pred_classes = [ class_dict[".".join(
                    #            [ dom_dict[cath_code][j - 2] for j in range(self.label_columns[0], i + 1) ]
                    #        )]["name"]  
                    #            for i in self.label_columns]
                    #    if self.verbose: print (cath_code, pred_classes)
                    #else:
                    #pred_classes = [dom_dict[cath_code][i-2] for i in self.label_columns]
                    pred_classes = []
                    if self.dataset in ['cath']:
                        for L in self.label_columns:
                            if L in [2,3,4,5]:
                                pred_classes.append( ".".join(
                                        [ dom_dict[cath_code][j - 2] for j in range(self.label_columns[0], L + 1) ]
                                    ) )
                            elif L in ['cath01_class', "cath02_architecture", "cath03_topology", "cath04_homologous_superfamily"]:
                                pred_classes.append( ".".join(
                                        [ dom_dict[cath_code][j - 2] for j in range(self.label_columns[0], L + 1) ]
                                    ) )
                            else:
                                pred_classes.append(dom_dict[cath_code][L - 2])
                                
                    if self.verbose: print (cath_code, pred_classes)#, [class_dict[p] for p in pred_classes[:3]])



                    if self.img_dim != None and self.img_size_bins == []:
                        if x.shape[1] != self.img_dim:
                            print ("Unexpected image size %s! Expected %s" % (x.shape, self.img_dim))
                            continue
                            # x=np.pad(x,(32,32,3),mode="constant", constant_values=0)
                        
                        assert x.shape[1] == self.img_dim
                    
                        img_data.append(x)
                        labels.append(str(cath_code))
                        classes.append(pred_classes)
                        orig_sizes.append(orig_size)
                    elif self.img_dim == None and self.img_size_bins == []:

                        
                    
                        img_data.append(x)
                        labels.append(str(cath_code))
                        classes.append(pred_classes)
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
                        labels_s[new_dim].append(cath_code)
                        classes_s[new_dim].append(pred_classes)

                    found_count += 1
                except IOError as ioe:
                    ioerror_list.append(f)
                    print(f, ioe)
                except ValueError as ve:
                    print(f, ve)
                    
            else:
                print("Warning: couldn't find image for ",cath_code)
                not_found_list.append(cath_code)
            MLhelpers.print_progress(fi, len(files))
        print ("Found %i image files." % len(img_data))
        
        with open("ioerrors.txt", "w") as iof:
            iof.write("\n".join(ioerror_list) + "\n")
        # print (np.array(img_data).reshape((len(img_data),32,32,3)))
        # print (len(img_data))
        
        if propose_image_bins:
            if self.img_dim != None and self.img_size_bins == []:
                # propose bins based on 1D clustering
                print("\n Propose %i k-means clusters over img size:" % (self.img_bin_clusters))

                size_df = pd.DataFrame(data={"img_sizes":orig_sizes}, index=labels)
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=self.img_bin_clusters)
                print (size_df["img_sizes"].values.reshape(-1, 1).shape)
                km.fit_transform(size_df["img_sizes"].values.reshape(-1, 1))
                
                size_df["img_sizes"].hist(bins=40)
                plt.xlabel("original image sizes")
                plt.ylabel("count")
                plt.title("%s" % (self.image_folder))
                plt.savefig("img_size_clusters00.png")
                
                size_df["clusters"] = km.labels_
                cl = size_df["clusters"].unique()
                print("clusters: name, n_members, min, mean, max")
                bins = []
                for l in range(len(cl)):
                    members = size_df[size_df["clusters"] == cl[l]]
                    cl_min = members["img_sizes"].values.min()
                    cl_max = members["img_sizes"].values.max()
                    cl_mean = members["img_sizes"].values.mean()
                    cl_size = members["img_sizes"].values.size
                    bins.append(cl_max)
                    print (cl[l], cl_size, cl_min, cl_mean, cl_max)
                    
                    plt.plot([cl_min, cl_max], [20, 20], marker="o", label="cluster_%i: #%i (%i--%.2f--%i)" % (cl[l], cl_size, cl_min, cl_mean, cl_max))
                plt.legend(loc=0)
                plt.savefig("img_size_clusters01.png")
                bins = sorted(bins)
                
                print("To use these bins: '--dim 0 --img_bins %s'" % (",".join(list(map(str, bins)))))
                
                print("\n Propose %i equally sized image-size bins:" % (self.img_bin_clusters))
                n_samples = size_df.shape[0]
                
                # propose bins based on equal size
                chunk_size = n_samples / self.img_bin_clusters
                
                size_df_sorted = size_df.sort_values(by="img_sizes")
                
                bin_labels = []
                for i in range(n_samples):
                    bin_labels.append(i / chunk_size)
                size_df_sorted["bin_label"] = bin_labels
                cl = size_df_sorted["bin_label"].unique()
                
                print("bins: name, n_members, min, mean, max")
                bins = []
                for l in range(len(cl)):
                    members = size_df_sorted[size_df_sorted["bin_label"] == cl[l]]
                    cl_min = members["img_sizes"].values.min()
                    cl_max = members["img_sizes"].values.max()
                    cl_mean = members["img_sizes"].values.mean()
                    cl_size = members["img_sizes"].values.size
                    bins.append(cl_max)
                    print (cl[l], cl_size, cl_min, cl_mean, cl_max)
                    
                    plt.plot([cl_min, cl_max], [10, 10], marker="x", label="bin_%i: #%i (%i--%.2f--%i)" % (cl[l], cl_size, cl_min, cl_mean, cl_max))
                plt.legend(loc=0)
                plt.savefig("img_size_clusters02.png")
                print("To use these bins: '--dim 0 --img_bins %s'" % (",".join(list(map(str, bins)))))
        if self.img_dim != None:
            self.X = np.array(img_data)
            self.Ys = list(np.array(classes).T)
            self.labels = labels
            self.not_found = not_found_list
        elif self.img_dim == None and self.img_size_bins == []:
            self.X = img_data
            self.Ys = list(np.array(classes).T)
            self.labels = labels
            self.not_found = not_found_list
        else:
            self.dX = {s:np.array(img_data_s[s]) for s in self.img_size_bins}
            self.dYs = {s:np.array(classes_s[s]).T for s in self.img_size_bins}
            self.dlabels = {s:labels_s[s] for s in self.img_size_bins}
            self.not_found = not_found_list
    
    def zero_pad_image(self, x, new_dim):
        # https://stackoverflow.com/questions/8561896/padding-an-image-for-use-in-wxpython
        # print (self.img_size_bins)
        #new_dim = self.img_size_bins[-1]
        old_dim = x.shape[1]
        
        # print ("Old dim: %i, new dim: %i"%(old_dim, new_dim))
        if old_dim <= new_dim:
            bg = np.zeros((new_dim, new_dim, self.nchannels), dtype=x.dtype)
            if K.image_data_format() == "channels_first":
                bg[:, :old_dim, :old_dim] = x
            else:
                bg[:old_dim, :old_dim, :] = x
            x = bg
        return x
    
    def prepare_dataset(self, show_classnames=False):
        
        if self.img_size_bins == []:
            if self.h5_input == "":
                self.getTrainTestSetMulti()
            else:
                self.getTrainTestSetMulti_h5()
            
            if self.show_images: self.plot_sample_images()
            
            
            if 11 in self.label_columns:
                #index_11 = self.label_columns.index(11)
                #print ("col 11 index: ",index_11)
                self.Ys[-2] = list(map(int, self.Ys[-2] ))
            
            self.nb_classes = []
            self.Ymats = []
            for y in self.label_columns:#range(len(self.Ys)):
                if not y in [2,3,4,5]: continue
                #if len(self.Ys) >= 4 and not y + 2 in self.label_columns: continue  # expecting the data set to have 4 classifcation tasks. Makes sure only the desired classifications are carried out. as defined by column_labels
                    
                Y = self.Ys[self.label_columns.index(y)]
                ser = pd.Series(Y)
                uniques = sorted(ser.unique())
                self.nb_classes.append(len(uniques))
                print ("\n %i classes" % self.nb_classes[-1])
                if show_classnames:
                    for u in uniques:
                        cnt = sum(ser == u)
                        print ("\tclass label '%s' count = %i" % (u, cnt))
                
                self.Ymats.append(pd.get_dummies(Y).values)

        else:
            if self.h5_input == "":
                self.getTrainTestSetMulti()
            else:
                self.getTrainTestSetMulti_h5()
            # dYs: dict of lists of np.arrays
            s_nonempty = []
            for si in range(len(self.img_size_bins)):
                s = self.img_size_bins[si]
                # print(s,len(dlabels[s]))
                if len(self.dlabels[s]) == 0:
                    del self.dX[s]
                    del self.dYs[s]
                    del self.dlabels[s]
                else:
                    s_nonempty.append(s)
            self.img_size_bins = s_nonempty
            # print(self.img_size_bins)
                
            self.X = self.dX[self.img_size_bins[0]]
            self.Ys = self.dYs[self.img_size_bins[0]]
            # total_Ys = [ [] for y in range(len(self.label_columns))]
            
            self.labels = self.dlabels[self.img_size_bins[0]]
            # print(X.shape,len(Ys),len(labels))
            
            self.dYmats = {}
            self.dnb_classes = {}
            total_Ys = [[] for y in range(len(self.label_columns))]
            self.sn = []
            for s in self.img_size_bins:
                print ("\n\nImage bin of sizes up to", s)
                
                if self.show_images: self.plot_sample_images()
                
                self.dnb_classes[s] = []
                for y in range(len(self.label_columns)):
                    
                    Y = self.dYs[s][y]
                    
                    print ("   Y shape:", Y.shape)
                    if len(total_Ys[y]) == 0:
                        total_Ys[y] = Y
                        # print ("Y",Y.shape)
                    else:
                        # print(total_Ys[y].shape,Y.shape)
                        total_Ys[y] = np.concatenate([total_Ys[y], Y], axis=0)
                    # print ("len(Y)",len(Y),len(total_Ys[y]))
                    
                    ser = pd.Series(Y)
                    uniques = sorted(ser.unique())
                    self.dnb_classes[s].append(len(uniques))
                    print ("\n %i classes" % self.dnb_classes[s][-1])
                    if show_classnames:
                        for u in uniques:
                            cnt = sum(ser == u)
                            print ("\tclass label '%s' count = %i" % (u, cnt))
                self.sn.append(len(Y))
            
            total_Y_mat = [pd.get_dummies(total_Ys[y]).values for y in range(len(self.label_columns))]
            
            # for y in range(len(self.label_columns)):
            #    print (total_Ys[y].shape)
            #    print (total_Y_mat[y].shape)
            self.nb_classes = []
            for si in range(len(self.img_size_bins)):  
                s = self.img_size_bins[si]
                self.dYmats[s] = [None for y in range(len(self.label_columns))]
                tmp_sn = [0] + self.sn
                # print (tmp_sn)
                for y in range(len(self.label_columns)):
                    ser = pd.Series(total_Ys[y])
                    uniques = sorted(ser.unique())
                    self.nb_classes.append(len(uniques))
                    # print(sum(tmp_sn[:si+1]),sum(tmp_sn[:si+2]))
                    self.dYmats[s][y] = total_Y_mat[y][sum(tmp_sn[:si + 1]):sum(tmp_sn[:si + 2])]
                    # print ("np.array(dYmats[s][y]).shape",s,y,dYmats[s][y].shape)
            self.Ymats = self.dYmats[self.img_size_bins[0]]
            # nb_classes=dnb_classes[self.img_size_bins[0]]

        if self.use_seq_encoder:
            self.prepare_sequence_data()
            print ("seq data shape:", self.seqdata.shape)
            for i in range(100):
                if self.labels[i] in self.seqdict:
                    print(len(self.seqdict), len(self.labels))
                    print (self.seqdict[self.labels[i]])
                    print(self.sequences[i])
                    if not self.use_biovec and self.seq_encoding=="onehot": 
                        print (seq_from_onehot(self.seqdata[i,]))
                    break
                else:
                    print ("Label not found in seqdict:",self.labels[i])

    def get_seq_from_pdb(self, cathid):
        
        loc = os.path.join( self.root_path, self.pdb_folder, "%s.pdb"%(cathid) )
        if os.path.exists( loc ):
            try:
                for altloc in ['ABCDEFG']:
                    sequence = prody.parsePDB(loc,altloc=altloc).select("protein and calpha").getSequence()
                    self.seq_maxlength = max(self.seq_maxlength, len(sequence))
                    if type(sequence)!=type(None): break
                
            except AttributeError as ae:
                sequence = ""
                print (ae, cathid)
            #counter += 1
            #print (counter, cathid, len(sequence), sequence)
            return sequence
            
        else:
            print("ERROR: couldn't find "+loc)
            return None
        
    def get_sequence_data(self, save_fasta=False):
        if self.seq_from_pdb:
            
            #counter = 0
            for cathid in self.labels:
                sequence = self.get_seq_from_pdb(cathid)
                if type(sequence) != type(None):
                    self.seqdict[cathid] = sequence
                self.sequences.append(sequence)
            if save_fasta:
                self.seqdict2fasta()
        else:
            if False:
                with open(os.path.join(self.root_path, "cath-domain-seqs.fa")) as fastafile:
                    for entry in fastafile.read().split(">")[1:]:
                        try:
                            tmp = entry.split("\n")
                            meta = tmp[0].strip()
                            # startstop = meta.split("/")[1].split("_") 
                            
                            cathid = meta.split("|")[2].split("/")[0]
                            sequence = tmp[1].strip()
                            self.seq_maxlength = max(self.seq_maxlength, len(sequence))
                            # print( cathid, startstop, sequence)
                            self.seqdict[cathid] = sequence
                        except ValueError as ve:
                            print (ve)
                            print (entry)
            else:
                with open(os.path.join(self.root_path, "seqdict_20798.fa")) as fastafile:
                    for entry in fastafile.read().split(">")[1:]:
                        try:
                            tmp = entry.split("\n")
                            
                            cathid = tmp[0].strip()
                            sequence = tmp[1].strip()
                            self.seq_maxlength = max(self.seq_maxlength, len(sequence))
                            # print( cathid, startstop, sequence)
                            self.seqdict[cathid] = sequence
                        except ValueError as ve:
                            print (ve)
                            print (entry)
        
    def prepare_sequence_data(self):
        print("Preparing sequence data...")
        if self.seqdict == {} :
            print( self.seqdict )
            self.get_sequence_data()
            assert self.seqdict != {}

        if self.use_biovec:
            print ("Using biovec...")
            self.seqdata = self.biovec_encoding()
        else:
            print("not using biovec...")
            if type(self.manual_seq_maxlength) != type(None):
                input_seqlength = self.manual_seq_maxlength
            else:
                input_seqlength = self.seq_maxlength

                
            print ("n sequences:", len(self.sequences))
            if len(self.sequences) == 0:
            
                for ci in range(len(self.labels)):
                    cath_code = self.labels[ci]
                    
                    
                    
                    seq = ""
                    if cath_code in self.seqdict:   
                        seq = self.seqdict[cath_code]
                        if "B" in seq: 
                            seq = seq.replace("B","X") # TODO look into this
                        
                    else:
                        print("ERROR: sequence not found in dict for ",cath_code)
                        seq = self.get_seq_from_pdb(cath_code)
                        if type(seq) != type(None):
                            
                            print("    SOLVED: obtained sequence from PDB file for ",cath_code)
                        else:
                            assert False, cath_code
                
                        
                    #print(cath_code, len(seq), self.Ys[-2][ci])
                    if len(seq) != int(self.Ys[-2][ci]):
                        print("sequence length mismatch:",cath_code)
                        seq = self.get_seq_from_pdb(cath_code)
                        self.Ys[-2][ci] = len(seq)
                        print(cath_code, len(seq), self.Ys[-2][ci])
                        assert int(len(seq)) == int(self.Ys[-2][ci])

                    self.sequences.append(seq)
                


            self.seqdata = self.sequence_encoding()
            if self.seq_encoding=='index' and self.seq_index_extra_dim:
                self.seqdata = self.seqdata.reshape( ( len(self.seqdata), input_seqlength, 1)  )

            print("seqdata.shape:",self.seqdata.shape)
    
    def load_model_from_files(self, modlabel_abspath):
        modlabel_abspath = os.path.abspath(modlabel_abspath)
        print("Loading model:", modlabel_abspath)
        self.modlabel = os.path.basename(modlabel_abspath)
        print("Label for model:",self.modlabel)
        self.model_arch_file = modlabel_abspath+"_model_arch.json"
        self.model_weights_file = modlabel_abspath+"_model_weights.hdf5"
        self.load_model = True
        self.keras_model = self.get_model(generate_modlabel=False)
        self.orig_model = self.keras_model
        inputs=self.keras_model.inputs
        input_shapes = {i.name:i.shape for i in inputs}
        print(input_shapes)
        if "seq2img" in self.modlabel: 
            self.model="seq2img"
            self.seq_encoder=True
            self.seq_decoder=False
            self.img_decoder=True
            self.img_encoder=False
            self.seq_enc_arch="lstm"
            self.use_embedding=False
            self.seq_padding="post"
            self.use_biovec=False

        for i in input_shapes:
            if "seq_input" in i:
                print("seq_input:",i,input_shapes[i])
                self.manual_seq_maxlength = input_shapes[i][1]
                sdim=input_shapes[i][2]
                
                if sdim==7: self.seq_encoding = "physchem"
                elif sdim>7: self.seq_encoding = "onehot"
                else: self.seq_encoding = "index"
                print(self.manual_seq_maxlength, self.seq_encoding, sdim)
            if "main_input" in i:
                print("main_input:",i,input_shapes[i])
                self.img_dim = input_shapes[i][1]
                self.nchannels = input_shapes[i][-1]
        
        
        outputs=self.keras_model.outputs
        output_shapes = {o.name.split("/")[0]:o.shape for o in outputs}
        
        self.label_columns = list(output_shapes.keys())
        print(self.label_columns)

    def aa2phys(self, aa_1_letter):
        return np.array([self.aa2phys_dict[aa_1_letter]['hp'],
                        self.aa2phys_dict[aa_1_letter]['mw'],
                        self.aa2phys_dict[aa_1_letter]['charge'],
                        self.aa2phys_dict[aa_1_letter]['N'],
                        self.aa2phys_dict[aa_1_letter]['S'],
                        self.aa2phys_dict[aa_1_letter]['O'],
                        self.aa2phys_dict[aa_1_letter]['ar']] )

    def sequence_encoding(self, num_words=21):
        if type(self.manual_seq_maxlength) != type(None):
            input_seqlength = self.manual_seq_maxlength
        else:
            if self.seq_maxlength ==-1: 
                self.seq_maxlength = max([len(s) for s in self.sequences])
            input_seqlength = self.seq_maxlength
        print ("max seq length:",input_seqlength)
        seqdata=[]
        if self.seq_encoding=="index":
            tk = Tokenizer(num_words=num_words, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                            lower=True, split='', char_level=True, oov_token=None)
            tk.fit_on_texts(self.sequences)
            seqdata = tk.texts_to_sequences(self.sequences)
            print ("\nWord counts:", tk.word_counts)
            print ("\nWord docs",   tk.word_docs)
            print ("\nWord index:", tk.word_index)
            print ("\nDoc count",   tk.document_count)
        elif self.seq_encoding=="onehot":
            chars = 'ACDEFGHIKLMNPQRSTVWYX'
            aa2num = {}
            for a in range(len(chars)):
                zeros = np.zeros(len(chars) + 1)
                # print ()
                zeros[1:] = np.array([1 if c == chars[a] else 0 for c in chars])
                aa2num[chars[a]] = zeros
                #print (a, chars[a], zeros)
            # aa2num = {chars[a]:np.zeros(len(chars)+1) for a in range(len(chars))}
            for seq in self.sequences:
                
                tmparr = np.zeros((min(input_seqlength, len(seq)), len(chars) + 1))
                tmparr[:, 0] = 1.0
                tmparr[0:min(len(seq), input_seqlength), ] = np.array([aa2num[seq[l]] for l in range(min(len(seq), input_seqlength))])
                #print(seq)
                #for row in tmparr:
                #    print("".join(map(str,row)))
                seqdata.append(tmparr)
        elif self.seq_encoding=="physchem":
            for seq in self.sequences:
                tmparr = np.array( [self.aa2phys( seq[l] ) for l in range(min(len(seq), input_seqlength))] )
                seqdata.append(tmparr)
        
        print ("first element:", seqdata[0])
        print(input_seqlength,self.seq_padding,self.pad_value)
        seqdata = sequence.pad_sequences(seqdata,
                                        maxlen=input_seqlength, 
                                        dtype=np.float32, 
                                        padding=self.seq_padding, 
                                        truncating=self.seq_padding
                                        ,value=self.pad_value
                                        )
        
        print ("first element:", seqdata[0])
        
        return seqdata
    
    def randomly_blank_images(self, fraction=0.1):
        if fraction > 0.0:
            shape = self.X.shape[1:]
            blank = np.zeros(shape)
            blank_ids = random.sample(range(len(self.labels)), int(len(self.labels)*fraction))
            print ("Blanking %i images."%len(blank_ids))
            for i in blank_ids:
                self.X[i,] = blank
            print(self.X.shape)
            print(self.X[blank_ids[0],:10,:10,0])
            print(self.X[blank_ids[0]+1,:10,:10,0])
            return blank_ids
        else:
            return []

    def randomly_blank_sequences(self, fraction=0.1):
        blank_ids = random.sample(range(len(self.labels)), len(self.labels)*fraction)
        # TODO :::
        return blank_ids

    def biovec_encoding(self):
        if self.generate_protvec_from_seqdict:
            with open( os.path.join(self.root_path, self.input_fasta_file),"w" ) as fa:
                for cathid in sorted(self.seqdict.keys()):
                    fa.write(">%s\n%s\n"%(cathid, self.seqdict[cathid]))
            assert os.path.exists(os.path.join(self.root_path, self.input_fasta_file))
            self.pv = biovec.ProtVec(fasta_fname=os.path.join(self.root_path, self.input_fasta_file), #fasta file for corpus
                                corpus=None, #corpus object implemented by gensim
                                n=self.biovec_ngram, #n of n-gram
                                size=self.biovec_length, 
                                corpus_fname=os.path.join(self.root_path, self.input_fasta_file.split(".")[0]+"_corpus.txt"),  
                                sg=1, 
                                window=25, 
                                min_count=1, #least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
                                workers=8
                                )
            self.pv.save(os.path.join(self.root_path, self.input_fasta_file.split(".")[0]+"_protvec.model"))
            self.biovec_model = os.path.join(self.root_path, self.input_fasta_file.split(".")[0]+"_protvec.model")
            print("New biovec model saved to file:",self.biovec_model)

        else:
            print ("Loading biovec model from file:",self.biovec_model)
            assert os.path.exists(self.biovec_model)
            self.pv = biovec.models.load_protvec(self.biovec_model)

            
        seq_enc = []
    
        for seq in self.sequences:
                
            aa_valididity = [x in "ACDEFGHIKLMNPQRSTVWYX" for x in seq]
            if not all(aa_valididity):
                if "B" in seq: seq = seq.replace("B","X") # TODO look into this
                if "X" in seq: seq = seq.replace("X","A") # TODO look into this
                aa_valididity = [x in "ACDEFGHIKLMNPQRSTVWYX" for x in seq]
                assert all(aa_valididity)," : \n%s\n%s"%(seq, "".join(map(str,map(np.int8,aa_valididity)))) 
            pvec = np.transpose(np.array(self.pv.to_vecs(seq)))
            #print ("PVEC:",pvec.shape)
            if self.biovec_flatten:
                seq_enc.append(pvec.reshape( (self.biovec_length*3, 1) ))
            else:
                seq_enc.append(pvec)
            # else:
            #     print("ERROR: cath code not found in seqdict:",cath_code)
            #     if self.biovec_flatten:
            #         tmparr = np.zeros((self.biovec_length*3,1))
            #     else:
            #         tmparr = np.zeros((self.biovec_length,3))
                
                
            #seq_enc.append(tmparr)
            #print("ERROR: could not find CATH code in self.labels ",cath_code)
            #self.seqdata = np.array(seq_enc)
        return np.array(seq_enc)

    def seqdict2fasta(self, outpath="", overwrite=False):
        if len(self.seqdict)!=0:

            if outpath=="":
                outpath = os.path.join(self.root_path, "seqdict_%i.fa"%(len(self.seqdict)))
            if not os.path.exists(outpath) or overwrite:
                with open(outpath, "w") as f:
                    f.write("\n".join([">%s\n%s"%(x, self.seqdict[x]) for x in self.seqdict]))
    
    def get_model(self, generate_modlabel=True, plot_model_diagram=True):
        
        if not self.load_model:
            if self.model == "fullconv":
                my_model = self.get_fullconv_model(branch_2d=self.use_img_encoder or self.use_img_decoder, branch_1d=self.use_seq_decoder or self.use_seq_encoder)
            elif self.model == "seq2img":
                my_model = self.get_seq2img_model2()
            elif self.model == "seq2img_gan":
                my_model =  self.get_seq2img_gan()
            elif self.model == "seq2img_gan_gen":
                self.get_generator(compile=False, add_outputs=True)
                my_model = self.generator
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
        else:
            print ("Loading model...")
            print (self.model_arch_file)
            print (self.model_weights_file)
            with open(self.model_arch_file) as arch:
                my_model = model_from_json(arch.read())
            my_model.load_weights(self.model_weights_file)
        
        if generate_modlabel:
            self.generate_modlabel()
            print ("Summary string for model:")
            print (self.modlabel)
        
        if plot_model_diagram:
            self.plot_model_graph(my_model)
        return my_model
    
    def plot_model_graph(self, my_model=None, rankdir='TB'):
        if type(my_model)==type(None):
            my_model = self.keras_model
        plot_model(my_model, to_file=os.path.join(self.outdir, self.modlabel+'_model-diagram.png'),show_shapes=True, show_layer_names=True, rankdir=rankdir)#'LR'

    def generate_modlabel(self):
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
        if "cath01_class" in self.nb_classes: clf_reg_string += "C"
        if "cath02_architecture" in self.nb_classes: clf_reg_string += "A"
        if "cath03_topology" in self.nb_classes: clf_reg_string += "T"
        if "cath04_homologous_superfamily" in self.nb_classes: clf_reg_string += "H"
        if "sequence_length" in self.nb_classes: clf_reg_string += "L"

        
        self.modlabel = "%s_%s%s_%s_Dr%s_%s_pt%i_%sx%sx%i_%i_bt%i_ep%i_f%ix%i_fc%ix%i_%s_%iG" % (
                            self.generic_label, self.model,
                            arch_string,
                            self.batch_norm_order,
                            str(self.dropout),
                            self.act,
                            int(self.use_pretrained),
                            str(self.img_dim), str(self.img_dim), self.nchannels, len(self.labels),
                            self.batch_size, self.epochs, self.kernel_shape[0], self.kernel_shape[1],
                            self.fc1, self.fc2, clf_reg_string, self.ngpus
                        )
    
    def dense_block(self, x, n_neurons, n_layers=2, act='default', init='he_normal', layer_order="LAB", names=False):
        if act=='default': act = self.act
        if names==False: names = [False for _ in range(n_layers)]
        assert len(names)==n_layers
        assert layer_order in ["LAB", "BLA", "LBA", "LA"]
        if not self.batch_norm: layer_order="LA"
        for i in range(n_layers):

            if layer_order=="BLA":
                x = BatchNormalization(name=names[i]+"bn"+str(i))( x )
            
            x = Dense(n_neurons, kernel_initializer = init, name=names[i]+"dense"+str(i))(x)
            
            if layer_order=="LBA":
                x = BatchNormalization(name=names[i]+"bn"+str(i))( x )
            
               
            if layer_order=="LAB":   
                x = Activation(act, name=names[i]+act+str(i))(x)
                x = BatchNormalization(name=names[i])( x ) 
            else:
                x = Activation(act, name=names[i])(x)
            
        return x
    
    def conv_block(self, x, dim, n_filters, n_layers=2, act='default', init='he_normal', layer_order="LAB", names=[0,0], filter_shape=None):
        if act=='default': act = self.act
        if names==False: names = [False for _ in range(n_layers)]
        assert len(names)>=n_layers
        assert layer_order in ["LAB", "BLA", "LBA", "LA"]
        if not self.batch_norm: layer_order="LA"
        assert dim in [1,2,3]
        for i in range(n_layers):

            if layer_order=="BLA":
                x = BatchNormalization()( x )
            
            if dim==1:
                x = Conv1D(n_filters, self.conv_filter_1d if type(filter_shape)==type(None) else filter_shape, padding = 'same', kernel_initializer = init)(x)
            elif dim==2:
                x = Conv2D(n_filters, self.conv_filter_2d if type(filter_shape)==type(None) else filter_shape, padding = 'same', kernel_initializer = init)(x)
            elif dim==3:
                x = Conv3D(n_filters, self.conv_filter_3d if type(filter_shape)==type(None) else filter_shape, padding = 'same', kernel_initializer = init)(x)
            
            if layer_order=="LBA":
                x = BatchNormalization()( x )
               
            if layer_order=="LAB":   
                x = Activation(act)(x)
                x = BatchNormalization(name=names[i])( x ) 
            else:
                x = Activation(act, name=names[i])(x)
            
        return x
    
    def attention_3d_block(self, inputs, seq_length, SINGLE_ATTENTION_VECTOR=False):
        # Thanks, Sean Grullon - https://tfs.gsk.com/tfs/PlatformProduct/DataScience/_git/Meganuclease?path=%2FModel_Training%2Ftrain_parallel_lstm_attention.py&version=GBmaster&_a=contents
        # original post: https://github.com/philipperemy/keras-attention-mechanism

        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, seq_length))(a) # this line is not useful. It's just to know which dimension is what.
        a = Dense(seq_length, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul
    
    def seq_input(self, input_seqlength):
 
        if self.use_biovec:
            if self.biovec_flatten:
                seq_input = Input(shape=(self.biovec_length*3, 1), dtype='float32', name='seq_input')
            else:
                seq_input = Input(shape=(self.biovec_length, 3), dtype='float32', name='seq_input')
        else:
            if self.seq_encoding=="onehot":
                seq_input = Input(shape=(input_seqlength, 22), dtype='float32', name='seq_input')
            elif self.seq_encoding=="physchem":
                seq_input = Input(shape=(input_seqlength, 7), dtype='float32', name='seq_input')
            else:
                if self.seq_enc_arch in ['cnn'] or not self.use_embedding:
                    seq_input = Input(shape=(input_seqlength, 1), dtype='float32', name='seq_input')
                else:
                    seq_input = Input(shape=(input_seqlength, ), dtype='float32', name='seq_input')
        return seq_input
        
    def conv_block_img_encoder(self, input, img_filters, n_layers, spatial_dropout=True):
        conv = self.conv_block(input, 2, img_filters, n_layers=n_layers, act=self.act, init='he_normal', layer_order=self.batch_norm_order, filter_shape=self.conv_filter_2d)
        if spatial_dropout: conv = SpatialDropout2D(self.dropout)(conv)
        
        return conv
    
    def conv_block_seq_encoder(self, input, seq_filters, n_layers, spatial_dropout=True,small_filter=False):
        conv = self.conv_block(input, 1, seq_filters, n_layers=n_layers, act=self.act, init='he_normal', layer_order=self.batch_norm_order, filter_shape=self.conv_filter_1d)
        if spatial_dropout: conv = SpatialDropout1D(self.dropout)(conv)
        if small_filter:
            conv3 = self.conv_block(input, 1, seq_filters*2, n_layers=n_layers, act=self.act, init='he_normal', layer_order=self.batch_norm_order, filter_shape=3)
            if spatial_dropout: conv3 = SpatialDropout1D(self.dropout)(conv3)
            conv = concatenate([conv,conv3])
        return conv
        
    def conv_img_unet_encoder(self, input, img_start_filters):
        act_img_enc = self.act

       
        conv1 = self.conv_block(input, 2, img_start_filters, n_layers=2, act=act_img_enc, init='he_normal', layer_order=self.batch_norm_order)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 112

        conv2 = self.conv_block(pool1, 2, img_start_filters*2, n_layers=2, act=act_img_enc, init='he_normal', layer_order=self.batch_norm_order)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 56

        conv3 = self.conv_block(pool2, 2, img_start_filters*4, n_layers=2, act=act_img_enc, init='he_normal', layer_order=self.batch_norm_order)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 28

        conv4 = self.conv_block(pool3, 2, img_start_filters*8, n_layers=2, act=act_img_enc, init='he_normal', layer_order=self.batch_norm_order)
        drop4 = Dropout(self.dropout)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # 14

        conv5 = self.conv_block(pool4, 2, img_start_filters*16, n_layers=2, act=act_img_enc, init='he_normal', layer_order=self.batch_norm_order, names=[0, "img_conv_bottleneck"])
        drop5 = Dropout(self.dropout)(conv5)

        return drop5, drop4, conv3, conv2, conv1

    def conv_img_unet_decoder(self, drop5, drop4, conv3, conv2, conv1, img_start_filters):
        act_img_dec = self.act

        up6 = UpSampling2D(size = (2,2))(drop5)
        up6 = Conv2D(img_start_filters*8, 2, padding = 'same', kernel_initializer = 'he_normal', activation=act_img_dec)(up6)
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = self.conv_block(merge6, 2, img_start_filters*8, n_layers=2, act=act_img_dec, init='he_normal', layer_order=self.batch_norm_order)

        up7 = UpSampling2D(size = (2,2))(conv6)
        up7 = Conv2D(img_start_filters*4, 2, padding = 'same', kernel_initializer = 'he_normal', activation=act_img_dec)(up7)
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = self.conv_block(merge7, 2, img_start_filters*4, n_layers=2, act=act_img_dec, init='he_normal', layer_order=self.batch_norm_order)

        up8 = UpSampling2D(size = (2,2))(conv7)
        up8 = Conv2D(img_start_filters*2, 2, padding = 'same', kernel_initializer = 'he_normal', activation=act_img_dec)(up8)
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = self.conv_block(merge8, 2, img_start_filters*2, n_layers=2, act=act_img_dec, init='he_normal', layer_order=self.batch_norm_order)

        up9 = UpSampling2D(size = (2,2))(conv8)
        up9 = Conv2D(img_start_filters, 2, padding = 'same', kernel_initializer = 'he_normal', activation=act_img_dec)(up9)
        merge9 = concatenate([conv1,up9], axis = 3)
        
        conv9 = self.conv_block(merge9, 2, img_start_filters, n_layers=2, act=act_img_dec, init='he_normal', layer_order=self.batch_norm_order)
        
        conv9 = Conv2D(6, self.conv_filter_2d, padding = 'same', kernel_initializer = 'he_normal', activation=act_img_dec)(conv9)
        img_activation_out = Activation('linear', name="img_decoder")( Conv2D(self.nchannels, 1)(conv9) )

        return img_activation_out
    
    def conv_seq_unet_encoder(self, seq_input, seq_start_filters):
        act_seq_enc = self.act
        seq_conv1 = self.conv_block(seq_input, 1, seq_start_filters, n_layers=2, act=act_seq_enc, init='he_normal', layer_order=self.batch_norm_order)
        seq_pool1 = MaxPooling1D(pool_size=2)(seq_conv1) # 800 / 64

        seq_conv2 = self.conv_block(seq_pool1, 1, seq_start_filters*2, n_layers=2, act=act_seq_enc, init='he_normal', layer_order=self.batch_norm_order)
        seq_pool2 = MaxPooling1D(pool_size=2)(seq_conv2) # 400 / 32

        seq_conv3 = self.conv_block(seq_pool2, 1, seq_start_filters*4, n_layers=2, act=act_seq_enc, init='he_normal', layer_order=self.batch_norm_order)
        seq_pool3 = MaxPooling1D(pool_size=2)(seq_conv3) # 200 / 16

        seq_conv4 = self.conv_block(seq_pool3, 1, seq_start_filters*8, n_layers=2, act=act_seq_enc, init='he_normal', layer_order=self.batch_norm_order)
        seq_drop4 = Dropout(self.dropout)(seq_conv4)
        seq_pool4 = MaxPooling1D(pool_size=2)(seq_drop4) # 100 / 8

        seq_conv5 = self.conv_block(seq_pool4, 1, seq_start_filters*16, n_layers=2, act=act_seq_enc, init='he_normal', layer_order=self.batch_norm_order)
        seq_drop5 = Dropout(self.dropout)(seq_conv5)

        return seq_drop5, seq_drop4, seq_conv3, seq_conv2, seq_conv1

    def conv_seq_unet_decoder(self, seq_drop5, seq_drop4, seq_conv3, seq_conv2, seq_conv1, seq_start_filters):
        act_seq_dec = self.act
        seq_up6 = UpSampling1D(size = 2)(seq_drop5)
        seq_up6 = Conv1D(seq_start_filters*8, 2, padding = 'same', kernel_initializer = 'he_normal', activation=act_seq_dec)( seq_up6 )  # 200 / 12
        seq_merge6 = concatenate([seq_drop4,seq_up6], axis = 2)
        seq_conv6 = self.conv_block(seq_merge6, 1, seq_start_filters*8, n_layers=2, act=act_seq_dec, init='he_normal', layer_order=self.batch_norm_order)

        seq_up7 = UpSampling1D(size = 2)(seq_conv6)
        seq_up7 = Conv1D(seq_start_filters*4, 2, padding = 'same', kernel_initializer = 'he_normal', activation=act_seq_dec)( seq_up7 )  # 400 / 24
        seq_merge7 = concatenate([seq_conv3,seq_up7], axis = 2)
        seq_conv7 = self.conv_block(seq_merge7, 1, seq_start_filters*4, n_layers=2, act=act_seq_dec, init='he_normal', layer_order=self.batch_norm_order)

        seq_up8 = UpSampling1D(size = 2)(seq_conv7)
        seq_up8 = Conv1D(seq_start_filters*2, 2, padding = 'same', kernel_initializer = 'he_normal', activation=act_seq_dec)( seq_up8 )
        seq_merge8 = concatenate([seq_conv2,seq_up8], axis = 2)
        seq_conv8 = self.conv_block(seq_merge8, 1, seq_start_filters*2, n_layers=2, act=act_seq_dec, init='he_normal', layer_order=self.batch_norm_order)

        seq_up9 = UpSampling1D(size = 2)(seq_conv8)
        seq_up9 = Conv1D(seq_start_filters, 2, padding = 'same', kernel_initializer = 'he_normal', activation=act_seq_dec)( seq_up9 )
        seq_merge9 = concatenate([seq_conv1,seq_up9], axis = 2)
        seq_conv9 = self.conv_block(seq_merge9, 1, seq_start_filters, n_layers=2, act=act_seq_dec, init='he_normal', layer_order=self.batch_norm_order)

        seq_conv9 = Conv1D(32, self.conv_filter_1d, padding = 'same', kernel_initializer = 'he_normal', activation=act_seq_dec)(seq_conv9)

        # reconstructed output
        if self.use_biovec:
            if self.biovec_flatten:
                seq_activation_out = Activation('linear', name="seq_dec")(Conv1D(1, 1, activation='linear')(seq_conv9) ) 
            else:
                seq_activation_out = Activation('linear', name="seq_dec")(Conv1D(3, 1, activation='linear')(seq_conv9) ) 
        else:
            if self.seq_encoding=='index':
                seq_activation_out = Activation('linear', name="seq_dec")(Conv1D(1, 1, activation='linear')(seq_conv9) )
            else:
                seq_conv10 = Conv1D(22, 1   , activation=act_seq_dec)(seq_conv9) 
                seq_activation_out = Activation(self.softMaxAxis(2), name="seq_decoder")(seq_conv10)
        return seq_activation_out

    def rec_seq_encoder(self, seq_input, input_seqlength):
        #def lstm_seq_encoder(self, act='relu', hidden_length=128):
        print ("LSTM ENCODER")
        #lstm_input = Input(shape=(self.manual_seq_maxlength,22,), dtype='float32', name='lstm_input')
        print ("lstm input:",seq_input.shape)

        # masking
        seq_input = Masking(mask_value=np.nan )(seq_input)

        if not self.use_biovec and self.use_embedding:
            if self.seq_encoding=="onehot":
                lstm_emb = Embedding(22, self.lstm_enc_length)(seq_input)
            else:
                lstm_emb = Embedding(21, 2)(seq_input)
        else:
            lstm_emb = seq_input
        
        if self.use_lstm_attention:
            lstm_emb = self.attention_3d_block(lstm_emb, input_seqlength)

        #lstm_emb = BatchNormalization()(lstm_emb)
        ("lstm emb:",lstm_emb.shape)
        if self.seq_enc_arch in ['lstm']: 
            lstm_enc = CuDNNLSTM(self.lstm_enc_length, 
                            kernel_initializer='he_normal', 
                            recurrent_initializer='orthogonal', 
                            bias_initializer='zeros', 
                            unit_forget_bias=True, 
                            kernel_regularizer=None,#l2(0.1), 
                            recurrent_regularizer=None,#l2(0.1), 
                            bias_regularizer=None, 
                            activity_regularizer=l2(0.0001), 
                            kernel_constraint=None, 
                            recurrent_constraint=None, 
                            bias_constraint=None, 
                            return_sequences=True, 
                            return_state=False, 
                            stateful=False)(lstm_emb)
        elif self.seq_enc_arch=='bilstm':
            lstm_enc = Bidirectional(
                            CuDNNLSTM(self.lstm_enc_length, 
                                kernel_initializer='he_normal', 
                                recurrent_initializer='orthogonal', 
                                bias_initializer='zeros', 
                                unit_forget_bias=True, 
                                kernel_regularizer=None,#l2(0.1), 
                                recurrent_regularizer=None,#l2(0.1), 
                                bias_regularizer=None, 
                                activity_regularizer=l2(0.0001), 
                                kernel_constraint=None, 
                                recurrent_constraint=None, 
                                bias_constraint=None, 
                                return_sequences=True, 
                                return_state=False, 
                                stateful=False)
                                ,merge_mode='concat' )(lstm_emb)
        elif self.seq_enc_arch=='gru': 
            lstm_enc = Bidirectional(CuDNNGRU(self.lstm_enc_length*2, 
                            kernel_initializer='he_normal', 
                            recurrent_initializer='orthogonal', 
                            bias_initializer='zeros', 
                            activity_regularizer=None,#l2(0.0001), 
                            return_sequences=True, 
                            return_state=False),merge_mode='concat' )(lstm_emb)
            lstm_enc = BatchNormalization()(lstm_enc)
            lstm_enc = Dropout(self.dropout)(lstm_enc)
            lstm_enc = Bidirectional(CuDNNGRU(self.lstm_enc_length, 
                            kernel_initializer='he_normal', 
                            recurrent_initializer='orthogonal', 
                            bias_initializer='zeros', 
                            activity_regularizer=None,#l2(0.0001), 
                            return_sequences=True, 
                            return_state=False),merge_mode='concat' )(lstm_enc)
            lstm_enc = BatchNormalization()(lstm_enc)
            lstm_enc = Dropout(self.dropout)(lstm_enc)
        # lstm_enc = LSTM(self.lstm_enc_length,
        #         activation=act_seq_enc, 
        #         recurrent_activation='hard_sigmoid', 
        #         use_bias=True, 
        #         kernel_initializer='glorot_uniform', 
        #         recurrent_initializer='orthogonal', 
        #         bias_initializer='zeros', 
        #         unit_forget_bias=True, 
        #         kernel_regularizer=None, 
        #         recurrent_regularizer=None, 
        #         bias_regularizer=None, 
        #         activity_regularizer=None, 
        #         kernel_constraint=None, 
        #         recurrent_constraint=None, 
        #         bias_constraint=None, 
        #         dropout=0.0, 
        #         recurrent_dropout=0.0, 
        #         implementation=1, 
        #         return_sequences=True, 
        #         return_state=False, 
        #         go_backwards=False, 
        #         stateful=False, 
        #         unroll=False)(seq_input)
        
        
        #seq_drop5 = concatenate([GlobalAveragePooling1D()(lstm_enc[0]), lstm_enc[1]  ])
        seq_drop5 = lstm_enc#GlobalAveragePooling1D()(lstm_enc)
        #seq_drop5 = Dropout(self.dropout)(lstm_enc)
        #seq_drop5 = Dropout(0.25)(lstm_enc)
        
        
        
        #lstm_enc = LSTM(self.lstm_enc_length, activation=act_seq_enc, return_sequences=True)(seq_input)
        #print ("lstm enc:",lstm_enc.shape)

        #return lstm_enc, lstm_input
        return seq_drop5
    
    def rec_seq_decoder(self, seq_drop5): #TODO: unfinished
        #def lstm_seq_decoder(self, encoded, act='relu'):
        print ("lstm_seq_decoder: encoded",seq_drop5.shape)
        seq_activation_out = LSTM(self.lstm_dec_length, activation=self.act,return_sequences=True, name="seq_decoder")(seq_drop5)
        print ("lstm_seq_decoder: decoded",seq_activation_out.shape)
        #return lstm_dec
        #if self.use_biovec:
        #    if self.biovec_flatten:
        #        seq_activation_out = Activation('linear', name="seq_dec")(Conv1D(1, 1, activation='linear')(lstm_dec) ) 
        #    else:
        #        seq_activation_out = Activation('linear', name="seq_dec")(Conv1D(3, 1, activation='linear')(lstm_dec) ) 
        return seq_activation_out

    def trans_seq_encoder(self, seq_input, input_seqlength, transformer_depth):
        #https://github.com/kpot/keras-transformer
        from keras_transformer.transformer import TransformerBlock
        from keras_transformer.position import TransformerCoordinateEmbedding
        transformer_block = TransformerBlock(
            name='transformer',
            num_heads=8,
            residual_dropout=0.1,
            attention_dropout=0.1,
            use_masking=True)
        add_coordinate_embedding = TransformerCoordinateEmbedding(
            transformer_depth,
            name='coordinate_embedding')
        seq_input = Permute([2,1])(seq_input)
        print(seq_input.shape)
        #seq_input = Embedding(None, 8)(seq_input)
        #print(seq_input.shape)  
        output = seq_input # shape: (<batch size>, <sequence length>, <input size>)
        for step in range(transformer_depth):
            output = transformer_block(
                add_coordinate_embedding(output, step=step))
        return output
    
    def img_fullconv_clf_linker(self, x, n_layers=2):
        shape = x._keras_shape
        dim = shape[1]
        assert dim%2==0
        n_filters=shape[-1]
        while dim>1:
            print( dim, n_filters )
            # only even dims allowed

            x = AveragePooling2D(pool_size=(2, 2))(x) 
            n_filters = max(self.fullconv_img_clf_length, int(n_filters/2))
            x = self.conv_block(x, 2, n_filters, n_layers=n_layers, act=self.act, init='he_normal', layer_order=self.batch_norm_order)
            x = SpatialDropout2D(self.conv_dropout)(x)
            shape = x._keras_shape
            dim = shape[1]
            n_filters=shape[-1]
        return x
    
    def seq_fullconv_clf_linker(self, seq_drop5):
        #branch to classification
        seq_pool_down = MaxPooling1D(pool_size=2)(seq_drop5) # 7 / 4
        seq_conv_down = self.conv_block(seq_pool_down, 1, self.fullconv_seq_clf_length, n_layers=2, act=self.act, init='he_normal', layer_order=self.batch_norm_order)
        seq_drop_down = SpatialDropout1D(self.conv_dropout)(seq_conv_down)
        return seq_drop_down

    def get_seq2img_model(self):
        inputs=[]
        if self.manual_seq_maxlength != None:
            input_seqlength = self.manual_seq_maxlength
        else:
            input_seqlength = self.seq_maxlength
        print ("input_seqlength=", input_seqlength)
        seq_input = self.seq_input(input_seqlength)
        print ("seqinput", seq_input.shape)
        inputs.append(seq_input)
        #seq_input = Masking(mask_value=np.nan )(seq_input)
        #seq_input = Embedding()(seq_input)
        
        lstm_out, lstm_out2 = Bidirectional( CuDNNLSTM(self.lstm_enc_length, 
                            kernel_initializer='he_normal', 
                            recurrent_initializer='orthogonal', 
                            bias_initializer='zeros', 
                            unit_forget_bias=True, 
                            kernel_regularizer=None,#l2(0.1), 
                            recurrent_regularizer=None,#l2(0.1), 
                            bias_regularizer=None, 
                            activity_regularizer=None,#l2(0.0001), 
                            kernel_constraint=None, 
                            recurrent_constraint=None, 
                            bias_constraint=None, 
                            return_sequences=True, 
                            return_state=False, 
                            stateful=False), merge_mode=None)(seq_input)

        print("lstm:",lstm_out.shape, "lstm2:",lstm_out2.shape)
        lstm_out  = Dense(self.img_dim)(lstm_out) 
        lstm_out2 = Dense(self.img_dim)(lstm_out2)  
        print("lstm:",lstm_out.shape, "lstm2:",lstm_out2.shape)

        merge = dot( [lstm_out, lstm_out2 ], axes=1)
        #merge = BatchNormalization()(merge)
        merge = Activation('relu', name='self_merge_lstm')(merge)
        print(merge.shape)

        img_shape = tuple([merge._keras_shape[1], merge._keras_shape[2]] + [1] )
        img = Reshape(img_shape)(merge)
        
        drop5, drop4, conv3, conv2, conv1 = self.conv_img_unet_encoder(img, 64)
        img_activation_out = self.conv_img_unet_decoder(drop5, drop4, conv3, conv2, conv1, 64)
        #img_activation_out.name = "img_activation_out"
        print ("model inputs:", inputs)
        
        if not self.no_classifier:
            drop5 = self.img_fullconv_clf_linker(drop5, n_layers=1)#8x8
            


        outputs = self.add_cath_outputs_to_model(None if self.no_classifier else drop5, img_decoder=img_activation_out, seq_decoder=None)
        
        model = Model(inputs=inputs, outputs=outputs, name="seq2U")   

        return model
    
    def get_seq2img_model2(self):
        inputs=[]
        if self.manual_seq_maxlength != None:
            input_seqlength = self.manual_seq_maxlength
        else:
            input_seqlength = self.seq_maxlength
        print ("input_seqlength=", input_seqlength)
        seq_input = self.seq_input(input_seqlength)
        print ("seqinput", seq_input.shape)
        inputs.append(seq_input)
        #seq_input = Masking(mask_value=np.nan )(seq_input)
        #seq_input = Embedding()(seq_input)
        
        lstm_out = CuDNNLSTM(self.lstm_enc_length, 
                            kernel_initializer='he_normal', 
                            recurrent_initializer='orthogonal', 
                            bias_initializer='zeros', 
                            unit_forget_bias=True, 
                            kernel_regularizer=None,#l2(0.1), 
                            recurrent_regularizer=None,#l2(0.1), 
                            bias_regularizer=None, 
                            activity_regularizer=None,#l2(0.0001), 
                            kernel_constraint=None, 
                            recurrent_constraint=None, 
                            bias_constraint=None, 
                            return_sequences=True, 
                            return_state=False, 
                            stateful=False)(seq_input)

        print("lstm:",lstm_out.shape)
        #lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dense( int((self.img_dim*self.img_dim*self.nchannels) / input_seqlength) )(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        #lstm_out = TimeDistributed(Dense(self.img_dim*self.img_dim*self.nchannels)(lstm_out) )
        print("lstm:",lstm_out.shape)
        #img_shape = tuple([lstm_out._keras_shape[1], lstm_out._keras_shape[2]] + [self.nchannels] )
        lstm_out = Reshape((self.img_dim,self.img_dim,self.nchannels))(lstm_out)
        lstm_out = average([lstm_out, Permute([2,1,3])(lstm_out) ])
        lstm_out = LeakyReLU(alpha=0.2,name='self_merge_lstm')(lstm_out)
        print("lstm:",lstm_out.shape)
        drop5, drop4, conv3, conv2, conv1 = self.conv_img_unet_encoder(lstm_out, 64)
        img_activation_out = self.conv_img_unet_decoder(drop5, drop4, conv3, conv2, conv1, 64)
        #img_activation_out.name = "img_activation_out"
        print ("model inputs:", inputs)
        
        if not self.no_classifier:
            drop5 = self.img_fullconv_clf_linker(drop5, n_layers=1)#8x8

        outputs = self.add_cath_outputs_to_model(None if self.no_classifier else drop5, img_decoder=img_activation_out, seq_decoder=None)
        
        model = Model(inputs=inputs, outputs=outputs)   

        return model
    
    def get_generator(self, compile=True, add_outputs=False):
        inputs=[]
        if self.manual_seq_maxlength != None:
            input_seqlength = self.manual_seq_maxlength
        else:
            input_seqlength = self.seq_maxlength
        print ("input_seqlength=", input_seqlength)
        seq_input = self.seq_input(input_seqlength)
        print ("seqinput", seq_input.shape)
        inputs.append(seq_input)
        #seq_input = Masking(mask_value=np.nan )(seq_input)
        #seq_input = Embedding()(seq_input)
        
        lstm_out = Bidirectional( CuDNNLSTM(self.lstm_enc_length, 
                            kernel_initializer='he_normal', 
                            recurrent_initializer='orthogonal', 
                            bias_initializer='zeros', 
                            unit_forget_bias=True, 
                            kernel_regularizer=None,#l2(0.1), 
                            recurrent_regularizer=None,#l2(0.1), 
                            bias_regularizer=None, 
                            activity_regularizer=None,#l2(0.0001), 
                            kernel_constraint=None, 
                            recurrent_constraint=None, 
                            bias_constraint=None, 
                            return_sequences=True, 
                            return_state=False, 
                            stateful=False), merge_mode='concat')(seq_input)

        #print("lstm:",lstm_out.shape, "lstm2:",lstm_out2.shape)
        #lstm_out  = Dense(self.img_dim)(lstm_out) 
        #lstm_out2 = Dense(self.img_dim)(lstm_out2)  
        #print("lstm:",lstm_out.shape, "lstm2:",lstm_out2.shape)
        dot_layers = []
        for i in range(self.nchannels):
            d1  = Dense(self.img_dim)(lstm_out) 
            #d2 = Dense(self.img_dim)(lstm_out2)  
            dot_merge = Reshape([self.img_dim, self.img_dim, 1])(dot( [d1, d1 ], axes=1) )
            print(i,dot_merge.shape)
            dot_layers.append(dot_merge)
        #lstm_out2 = Dense(self.img_dim)(lstm_out)  
        #print("lstm:",lstm_out.shape, "lstm2:",lstm_out2.shape)
        if self.nchannels > 1:
            merge = concatenate(dot_layers, axis=3)
        else:
            merge = dot_layers[0]
        print("merge:",merge.shape)

        #merge = dot( [lstm_out, lstm_out2 ], axes=1)
        merge = BatchNormalization( )(merge)
        merge = Activation('relu', name='self_merge_lstm')(merge)
        print(merge.shape)

        #img_shape = tuple([merge._keras_shape[1], merge._keras_shape[2]] + [1] )
        #img = Reshape(img_shape)(merge)
        # if self.manual_seq_maxlength != None:
        #     input_seqlength = self.manual_seq_maxlength
        # else:
        #     input_seqlength = self.seq_maxlength
        # print ("input_seqlength=", input_seqlength)
        # seq_input = self.seq_input(input_seqlength)
        # print ("seqinput", seq_input.shape)
        # #seq_input = Masking(mask_value=np.nan )(seq_input)
        # #seq_input = Embedding()(seq_input)
        
        # lstm_out = Bidirectional( CuDNNLSTM(self.lstm_enc_length, 
        #                     kernel_initializer='he_normal', 
        #                     recurrent_initializer='orthogonal', 
        #                     bias_initializer='zeros', 
        #                     unit_forget_bias=True, 
        #                     kernel_regularizer=None,#l2(0.1), 
        #                     recurrent_regularizer=None,#l2(0.1), 
        #                     bias_regularizer=None, 
        #                     activity_regularizer=None,#l2(0.0001), 
        #                     kernel_constraint=None, 
        #                     recurrent_constraint=None, 
        #                     bias_constraint=None, 
        #                     return_sequences=True, 
        #                     return_state=False, 
        #                     stateful=False), merge_mode="concat")(seq_input)

        # print("lstm:",lstm_out.shape)
        # dot_layers = []
        # for i in range(self.nchannels):
        #     lstm_out  =lstm_out2= Dense(self.img_dim)(lstm_out) 
        #     dot_merge = Reshape([self.img_dim, self.img_dim, 1])(dot( [lstm_out, lstm_out2 ], axes=1) )
        #     print(i,dot_merge.shape)
        #     dot_layers.append(dot_merge)
        # #lstm_out2 = Dense(self.img_dim)(lstm_out)  
        # #print("lstm:",lstm_out.shape, "lstm2:",lstm_out2.shape)
        # if self.nchannels > 1:
        #     merge = concatenate(dot_layers, axis=3)
        # else:
        #     merge = dot_layers[0]
        # print("merge:",merge.shape)
        
        # #merge = BatchNormalization()(merge)
        
        # merge = Activation('relu', name='self_merge_lstm')(merge)
        # print(merge.shape)
        #s2u = get_seq2img_model()

        #tmp_model = Model(seq_input, merge)
        unet_name = 'unet_fullconv_imgED1.00_LAB_Dr0.0_relu_pt0_256x256x3_20798_bt48_ep100_f3x3_fc0x128_clfCATH_4G_split0.30_lo1.353_vlo1.101'
        with open(os.path.join("/hpc/ai_ml/ELT/results", unet_name+"_model_arch.json")) as arch:
            unet = model_from_json(arch.read())
        unet.load_weights(os.path.join("/hpc/ai_ml/ELT/results", unet_name+"_model_weights.hdf5") )
        

        #unet.layers.pop(0)
        #newnet = 
        
        #unet.get_input_at(0)(newnet.output)
        #out = unet(newnet.output)
        #
        #newnet = Model(merge, unet.ouputs)
        #print(newnet.summary())
        #print(newnet.get_layer(name='model_4').summary())

        #new_input = Input(tensor=merge,name='merge_input') 
        #unet(new_input )
        #unet.layers[0].input = merge
        # x = merge
        # print(x)
        # for l in unet.layers:
        #     print(l)
        #     x = l.input =(x)
        #print(unet.input  )
        #new_output = unet(new_input)
        #print( unet.summary() )
        if add_outputs:
            self.add_cath_outputs_to_model(None if self.no_classifier else unet.get_layer(name="img_conv_bottleneck").output, img_decoder=None, seq_decoder=None)

        self.generator = Model(seq_input, unet(merge), name="generator_G")#Model(seq_input, unet.output)
        print(self.generator.summary())
        print(self.generator.get_layer(name='model_4').summary())
        
        if compile:
            self.generator.compile(loss='mean_absolute_error', optimizer=self.get_optimizer())

        self.plot_model_graph(self.generator)

    def get_discriminator(self):
        input_tensor = Input(shape=(self.img_dim, self.img_dim, self.nchannels))
        my_model = DenseNet121(include_top=self.keras_top, weights="imagenet"if self.use_pretrained else None , input_tensor=input_tensor)
                
        print ("Layers from KERAS model:")
        print ([layer.name for layer in my_model.layers])
        # remove last output layer (prediction)
        #del my_model.layers[-5]
        print(my_model.layers[-1])
        # grab loose ends
        x = my_model.layers[-1].output

        # if self.crops_per_image>0:
        #     input_size = (self.crop_width, self.crop_height, self.nchannels)
        # else:
        #     input_size = (self.img_dim, self.img_dim, self.nchannels)
        # input = Input(input_size, dtype='float32', name='main_input') # 224
        # print ("Discriminator input shape:", input.shape)
        # drop5, drop4, conv3, conv2, conv1 = self.conv_img_unet_encoder(input, 64)
        # x = self.img_fullconv_clf_linker(drop5, n_layers=1)#8x8
        # if len(x._keras_shape)==4:
        #     x = GlobalAveragePooling2D()(x) 
        # elif len(x._keras_shape)==3:
        #     x = GlobalAveragePooling1D()(x) 
        output = Dense(1, activation='sigmoid',name="D_out")(x)
        
        self.discriminator = Model(input_tensor, output, name="discriminator_D")
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.get_optimizer())
        plot_model(self.discriminator, to_file=os.path.join(self.outdir, self.modlabel+'_discriminator-diagram.png'),show_shapes=True, show_layer_names=True, rankdir='TB')#'LR'
    
    def get_discriminator_from_G(self, G):
        print(G.get_layer(index=0))
        if self.crops_per_image>0:
            input_size = (self.crop_width, self.crop_height, self.nchannels)
        else:
            input_size = (self.img_dim, self.img_dim, self.nchannels)
        input = Input(input_size, dtype='float32', name='main_input_D') # 224
        # print ("Discriminator input shape:", input.shape)
        # drop5, drop4, conv3, conv2, conv1 = self.conv_img_unet_encoder(input, 64)
        
        print(G.summary())
        x = G.get_layer("model_4").get_layer("img_conv_bottleneck")
        print("img_conv_bottleneck",x.output.shape)
        if len(x.output.shape)==4:
            x = GlobalAveragePooling2D()(x.output) 
        elif len(x.output.shape)==3:
            x = GlobalAveragePooling1D()(x.output) 
        output = Dense(1, activation='sigmoid',name="D_out")(x)
        
        self.discriminator = Model(G.get_layer(name='model_4')(input), output, name="discriminator_D")
        print(self.discriminator.summary())
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.get_optimizer())
        plot_model(self.discriminator, to_file=os.path.join(self.outdir, self.modlabel+'_discriminator-diagram.png'),show_shapes=True, show_layer_names=True, rankdir='TB')#'LR'
    
    def get_seq2img_gan(self):
        self.get_generator()
        self.get_discriminator()#_from_G(self.generator)
        self.discriminator.trainable = False
        
        
        if self.manual_seq_maxlength != None:
            input_seqlength = self.manual_seq_maxlength
        else:
            input_seqlength = self.seq_maxlength
        print ("input_seqlength=", input_seqlength)
        seq_input = self.seq_input(input_seqlength)
        
        x = self.generator(seq_input)
        print("generator output:",x.shape)
        gan_output = self.discriminator(x)
        
        gan = Model(seq_input, gan_output, name="SEQ2IMG_GAN")
        
        return gan

    def train_gan(self):
        if self.ngpus == 1:
            self.keras_model = self.get_model()
        else:
            import tensorflow as tf

            with tf.device("/cpu:0"):
                self.orig_model = self.get_model()
                print ("Multi-GPU run - model on CPU:")
                print (self.orig_model)
                print(self.orig_model.summary())
                
            self.keras_model = multi_gpu_model(self.orig_model, gpus=self.ngpus)
        
        print (self.keras_model)
        print (self.keras_model.summary())
        #print (self.keras_model.get_config())
        
        self.keras_model.compile(loss='binary_crossentropy', optimizer=self.get_optimizer())
        
        train_img, val_img, train_seq, val_seq = train_test_split(self.X, self.seqdata, test_size = self.valsplit)
        batch_count = train_img.shape[0] / self.batch_size

        y_fake = np.zeros(self.batch_size) # 0-->fake
        y_real = np.full(self.batch_size, 0.9) # 0.9 --> real
 
        start = time.time()
        
        #print(input_dict)

        #print(output_dict)
        self.history = History()
        self.history.on_train_begin()
        self.history.history["g_loss"] = []
        self.history.history["d_loss"] = []
        self.plot_sample_images(indices=[1,10,100,1000])
        for e in range(1, self.epochs+1):
            print('-'*15, 'Epoch %d' % e, '-'*15)
            for b in range(int(batch_count)):
                
                image_batch = train_img[np.random.randint(0, train_img.shape[0], size=self.batch_size)]
                seq_batch = train_seq[np.random.randint(0, train_seq.shape[0], size=self.batch_size)]
                
                generated_images = self.generator.predict(seq_batch)

                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(image_batch, y_real)
                d_loss_fake = self.discriminator.train_on_batch(generated_images, y_fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                y_gen = np.ones(self.batch_size)
                self.discriminator.trainable = False
                g_loss = self.keras_model.train_on_batch(seq_batch, y_gen)
                print("batch %i/%i: d_loss_real = %f --- d_loss_fake = %f --- g_loss = %f"%(b+1,int(batch_count), d_loss_real, d_loss_fake, g_loss))
            self.history.history["d_loss"].append(d_loss)
            self.history.history["g_loss"].append(g_loss)
            self.history.epoch.append(e)
            if e == 1 or e % 20 == 0:
                self.plot_generated_images(e, self.generator,indices=[1,10,100,1000])
        
        
        print (len(self.history.history["loss"]), "epochs trained")
        for k in self.history.history.keys():
            print ("%s:\t%.3f" % (k, self.history.history[k][-1]))
        
        
        end = time.time()
        elapsed = end - start
        print ("Elapsed training time (s): %s"%str(elapsed))

        self.modlabel += "_split%.2f_lo%.3f_vlo%.3f" % (self.valsplit, self.history.history["loss"][-1], self.history.history["val_loss"][-1])
        # print self.history.history
        
        self.model_arch_file = os.path.join(self.outdir, "%s_model_arch.json" % self.modlabel)
        self.model_weights_file = os.path.join(self.outdir, "%s_model_weights.hdf5" % self.modlabel)
        
        if self.ngpus == 1:
            self.save_training_history()
            #with open(os.path.join(self.outdir, "%s_model_history.json" % self.modlabel), "w") as hist:
            #    hist.write(str(self.history.history))
            self.keras_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(self.keras_model.to_json())
            plot_model(self.keras_model, to_file=os.path.join(self.outdir, self.modlabel+'_model-diagram.png'),show_shapes=True, show_layer_names=True, rankdir='TB')#'LR'

        else:
            self.save_training_history()
            #with open(os.path.join(self.outdir, "%s_model_history.json" % self.modlabel), "w") as hist:
            #    hist.write(str(self.history.history))
            self.orig_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(self.orig_model.to_json())
            
            plot_model(self.orig_model, to_file=os.path.join(self.outdir, self.modlabel+'_model-diagram.png'),show_shapes=True, show_layer_names=True, rankdir='TB')#'LR'

        return self.modlabel  
    
    def plot_generated_images(self, epoch, generator, indices=[1,10,100,1000]):
        # show some random example images
        fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, squeeze=False)
        #randi = [random.randint(0, self.X.shape[0] - 1) for i in range(4)]
        test_imgs = [ np.squeeze(generator.predict(  self.seqdata[i,].reshape([1]+list(self.seqdata.shape[1:]))  ) ) for i in indices]
        titles = [self.labels[i] for i in indices]
        # test_img = X[0,:,:,:]
        print ("random test images:", test_imgs[0].shape)
        # if self.nchannels==1: 
        #    test_img=np.squeeze(test_img)
        #    print ("test image squeezed:",test_img.shape)
        # test_img *= test_img*255
        axes[0, 0].imshow(test_imgs[0],  interpolation="none");axes[0, 0].set_title(titles[0])
        axes[0, 1].imshow(test_imgs[1],  interpolation="none");axes[0, 1].set_title(titles[1])
        axes[1, 0].imshow(test_imgs[2],  interpolation="none");axes[1, 0].set_title(titles[2])
        axes[1, 1].imshow(test_imgs[3],  interpolation="none");axes[1, 1].set_title(titles[3])
        # le = LabelEncoder()
        plt.tight_layout()
        #plt.show()
        #print("writing: ",os.path.join(self.outdir, self.modlabel+"_random_images.png"))
        plt.savefig(os.path.join(self.outdir, self.modlabel+"_generated_images_ep%i.png"%epoch))
        plt.close('all')

    def get_fullconv_model(self, branch_2d=True, branch_1d=True):
        assert branch_2d or branch_1d
        inputs = []

        #PROTEIN STRUCTURE
        if branch_2d:
            img_start_filters = 64

            #PROTEIN STRUCTURE ENCODER
            if self.use_img_encoder:
                if self.crops_per_image>0:
                    input_size = (self.crop_width, self.crop_height, self.nchannels)
                else:
                    input_size = (self.img_dim, self.img_dim, self.nchannels)
                input = Input(input_size, dtype='float32', name='main_input') # 224
                inputs.append(input)
                drop5, drop4, conv3, conv2, conv1 = self.conv_img_unet_encoder(input,img_start_filters)
                if not self.fullconv_img_clf_length in [0, None]:
                    final_2d = self.img_fullconv_clf_linker(drop5)
                    final_pool = GlobalAveragePooling2D(name="final_2d")(final_2d) # latent space vector
                else:
                    final_2d = drop5
                    final_pool = GlobalAveragePooling2D(name="final_2d")(final_2d) # latent space vector

            #PROTEIN STRUCTURE DECODER
            if self.use_img_decoder:
                img_activation_out = self.conv_img_unet_decoder(drop5, drop4, conv3, conv2, conv1, img_start_filters)
                #img_activation_out.name = "img_activation_out"
            else:
                img_activation_out = None
        else:
            final_pool = None
            img_activation_out = None
        
        #PROTEIN SEQUENCE
        if branch_1d:
            seq_start_filters = 32

            # PROTEIN SEQUENCE ENCODER
            if self.use_seq_encoder:
                if self.manual_seq_maxlength != None:
                    input_seqlength = self.manual_seq_maxlength
                else:
                    input_seqlength = self.seq_maxlength
                print ("input_seqlength=", input_seqlength)
                seq_input = self.seq_input(input_seqlength)
                print ("seqinput", seq_input.shape)
                inputs.append(seq_input)

                if self.seq_enc_arch=='cnn':
                    seq_drop5, seq_drop4, seq_conv3, seq_conv2, seq_conv1 = self.conv_seq_unet_encoder(seq_input, seq_start_filters)
                    if not self.fullconv_seq_clf_length in [0, None]:
                        seq_final_pool = self.seq_fullconv_clf_linker(seq_drop5)
                        seq_final_pool = GlobalAveragePooling1D(name="final_1d")(seq_final_pool) # latent space vector

                    else:
                        seq_final_pool = GlobalAveragePooling1D(name="final_1d")(seq_drop5) # latent space vector
                elif self.seq_enc_arch=='single_conv':
                    seq_drop5 = self.conv_block_seq_encoder(seq_input, 128, 1, spatial_dropout=True)
                    #seq_drop4, seq_conv3, seq_conv2, seq_conv1 = None
                    if not self.fullconv_seq_clf_length in [0, None]:
                        seq_final_pool = self.seq_fullconv_clf_linker(seq_drop5)
                        seq_final_pool = GlobalAveragePooling1D(name="final_1d")(seq_final_pool) # latent space vector

                    else:
                        seq_final_pool = seq_drop5#GlobalAveragePooling1D(name="final_1d")(seq_drop5) # latent space vector
                elif self.seq_enc_arch in  ['lstm', 'gru', 'bilstm']:
                    seq_drop5 = self.rec_seq_encoder(seq_input, input_seqlength)
                    seq_final_pool = seq_drop5 # latent space vector
                elif self.seq_enc_arch=="trans":
                    seq_final_pool = self.trans_seq_encoder(seq_input, input_seqlength, 5)

                else:
                    seq_final_pool = seq_input
    	        
                
            #PROTEIN SEQUENCE DECODER
            if self.use_seq_decoder:
                if self.seq_dec_arch=='cnn':
                    seq_activation_out = self.conv_seq_unet_decoder(seq_drop5, seq_drop4, seq_conv3, seq_conv2, seq_conv1, seq_start_filters)
                    seq_activation_out.name = "seq_activation_out"
                   
                elif self.seq_dec_arch=='lstm':
                    seq_activation_out = self.rec_seq_decoder(seq_drop5)
                else:
                    seq_activation_out = None
            else:
                seq_activation_out = None
                
            
        else:
            seq_final_pool = None
            seq_activation_out=None
            
        
        # handle concatenated branches leading to classification task (if applicable). Use fully connected layers if fc1, fc2 arguments set
        if len(self.cath_column_labels) >0 and not self.no_classifier:

            if branch_1d and branch_2d:
                if self.merge_type == 'concatenate':
                    merge = concatenate([final_pool, GlobalAveragePooling1D(name="final_1d")(seq_final_pool)], axis=1, name='merge_seq_img')
                elif self.merge_type == 'average':
                    merge = average([final_pool, GlobalAveragePooling1D(name="final_1d")(seq_final_pool)], name='merge_seq_img')
                elif self.merge_type == 'dot':
                    img_shape = (final_2d._keras_shape[1]*final_2d._keras_shape[2], final_2d._keras_shape[3])
                    img = Reshape(img_shape)(final_2d)
                    merge = dot( [img, seq_final_pool ], axes=2, name='merge_seq_img')
                
                if self.classifier=='dense': # add dense layers before classification
                    final = self.add_fclayers_to_model(merge) 
                    if len(final._keras_shape)>2:
                        final = GlobalAveragePooling1D(name="final")(final)
                elif self.classifier=='conv':
                    shape = tuple(list(merge._keras_shape[1:])+[1])
                    reshape = Reshape(  shape )(merge)
                    final = self.add_conv_layers_to_model(  reshape  )
                    final = GlobalAveragePooling1D( name='final')(final)
                else:
                    final = Dropout(self.dropout, name='final')(merge)
            
            elif branch_2d and not branch_1d:
                if self.classifier=='dense':
                    final = self.add_fclayers_to_model(final_pool) 
                elif self.classifier=='conv':
                    shape = tuple(list(final_pool._keras_shape[1:])+[1])
                    reshape = Reshape(  shape )(final_pool)
                    final = self.add_conv_layers_to_model(  reshape  )
                    final = GlobalAveragePooling1D( name='final')(final)
                else:
                    
                    final = GlobalAveragePooling1D(name='final')(final_pool)
            
            elif not branch_2d and branch_1d:
                if self.classifier=='dense':
                    final = GlobalAveragePooling1D(name='final')(self.add_fclayers_to_model(seq_final_pool) )
                elif self.classifier=='conv':
                    shape = tuple(list(seq_final_pool._keras_shape[1:])+[1])
                    reshape = Reshape(  shape )(seq_final_pool)
                    final = self.add_conv_layers_to_model(  reshape  )
                    final = GlobalAveragePooling1D( name='final')(final)
                else:
                    
                    final = GlobalAveragePooling1D(name='final')(seq_final_pool)
            else:
                print("ERROR: enable at least one branch of the architecture!")
            
            print("final layer before classfication outputs:", final.shape, final.name)
        else:
            final = None


        print ("model inputs:", inputs)
        
        if type(self.label_columns) == list:
            if self.dataset=='cath':
                outputs = self.add_cath_outputs_to_model(final, img_decoder=img_activation_out, seq_decoder=seq_activation_out)
            else:
                outputs = [Dense(1,name='dummy')(final)]
        else:
            outputs = self.add_outputs_to_model(final, unet_decoder=img_activation_out, seqnet_decoder=seq_activation_out)
        
        
        
        model = Model(inputs=inputs, outputs=outputs)   
        # if branch_2d and not branch1d:
        #     model.get_layer("final_2d").name = "final"
        # elif not branch_2d and branch_1d:
        #     model.get_layer("final_1d").name = "final"
        return model
    
    def add_fclayers_to_model(self, x):
        
        if self.fc1 >0:
            x = self.dense_block(x, self.fc1 , 1,  layer_order=self.batch_norm_order, names=["fc_first"])
            x = Dropout(self.dropout)(x)
        if self.fc2 >0:
            x = self.dense_block(x, self.fc2 , 1, layer_order=self.batch_norm_order, names=["fc_last"])
            x = Dropout(self.dropout)(x)

        return x
    
    def add_conv_layers_to_model(self, x):
        if self.fc1 >0:
            x = self.conv_block(x, 1, self.fc1, n_layers=1, act=self.act, init='he_normal', layer_order=self.batch_norm_order, names=["fc_first"])
            x = Dropout(self.dropout)(x)
        if self.fc2 >0:
            x = self.conv_block(x, 1, self.fc2, n_layers=1, act=self.act, init='he_normal', layer_order=self.batch_norm_order, names=["fc_last"])
            x = Dropout(self.dropout)(x)
        return x
    
    def add_outputs_to_model(self, x, maxpool2D=None, unet_decoder=None, seqnet_decoder=None): # deprecated
        # x is the incoming layer from an existing model
        
        # collect outputs
        outputs = []
        if type(x)!=type(None):
            # attach four new output layers instead of removed output layer
            if 2 in self.label_columns: 
                Cout = Dense(self.nb_classes[0], activation="softmax", name="Cout")(x)
                outputs.append(Cout)
                
            if 3 in self.label_columns: 
                Aout = Dense(self.nb_classes[1], activation="softmax", name="Aout")(x)
                outputs.append(Aout)
                
            if 4 in self.label_columns: 
                Tout = Dense(self.nb_classes[2], activation="softmax", name="Tout")(x)
                outputs.append(Tout)
                
            if 5 in self.label_columns: 
                Hout = Dense(self.nb_classes[3], activation="softmax", name="Hout")(x)
                outputs.append(Hout)
            if 6 in self.label_columns: 
                S35out = Dense(self.nb_classes[4], activation="softmax", name="S35out")(x)
                outputs.append(S35out)
            if 7 in self.label_columns: 
                S60out = Dense(self.nb_classes[5], activation="softmax", name="S60out")(x)
                outputs.append(S60out)
            if 8 in self.label_columns: 
                S95out = Dense(self.nb_classes[6], activation="softmax", name="S95out")(x)
                outputs.append(S95out)
            if 9 in self.label_columns: 
                S100out = Dense(self.nb_classes[7], activation="softmax", name="S100out")(x)
                outputs.append(S100out)
            if 10 in self.label_columns: 
                NS100out = Dense(1, activation="linear", name="NS100out")(x)
                outputs.append(NS100out)
            if 11 in self.label_columns:
                LENout = Dense(1, activation="relu", name="LENout")(x)
                outputs.append(LENout)
            if 12 in self.label_columns:
                RESout = Dense(1, activation="linear", name="RESout")(x)
                outputs.append(RESout)
        
        # optionally, add a 2D decoder
        if self.use_img_decoder:
            
            dcdec = unet_decoder
            
            outputs.append(dcdec)
        
        if self.use_seq_decoder:
            seqdec = seqnet_decoder
            outputs.append( seqdec  )
        return outputs
    
    def add_cath_outputs_to_model(self, x, img_decoder=None, seq_decoder=None):
        # x is the incoming layer from an existing model


        # collect outputs
        outputs = []
        if type(x)!=type(None):
            if len(x._keras_shape)==4:
                x = GlobalAveragePooling2D()(x) 
            elif len(x._keras_shape)==3:
                x = GlobalAveragePooling1D()(x) 
            
            # attach four new output layers instead of removed output layer
            for l in self.label_columns:
                if l in ['cath01_class', "cath02_architecture", "cath03_topology", "cath04_homologous_superfamily",
                        "cath05_S35_cluster", "cath06_S60_cluster", "cath07_S95_cluster","cath08_S100_cluster"]:
                    outputs.append(Dense(self.nb_classes[l], activation="softmax", name=l)(x))

                elif l in ["sequence_length"]: 
                    outputs.append(Dense(1, activation="relu", name=l)(x))
            
        
        # optionally, add a 2D decoder
        if self.use_img_decoder:
            outputs.append(img_decoder)
        
        if self.use_seq_decoder:
            outputs.append( seq_decoder  )

        return outputs
    
    def softMaxAxis(self, axis):
        def soft(x):
            return softmax(x, axis=axis)
        return soft

    def get_output_labels(self): # deprecated
        my_outputs = []
        my_losses = {}
        my_loss_weights = {}
        my_metrics = {}
        # attach four new output layers instead of removed output layer
        if not self.no_classifier:
            if 2 in self.label_columns: 
                my_outputs.append("Cout")
                my_losses['Cout'] = 'categorical_crossentropy'
                my_loss_weights["Cout"] = self.cath_loss_weight
                my_metrics["Cout"] = 'accuracy'
            if 3 in self.label_columns: 
                my_outputs.append("Aout")
                my_losses['Aout'] = 'categorical_crossentropy'
                my_loss_weights["Aout"] = self.cath_loss_weight
                my_metrics["Aout"] = 'accuracy'
            if 4 in self.label_columns: 
                my_outputs.append("Tout")
                my_losses['Tout'] = 'categorical_crossentropy'
                my_loss_weights["Tout"] = self.cath_loss_weight
                my_metrics["Tout"] = 'accuracy'
            if 5 in self.label_columns: 
                my_outputs.append("Hout")
                my_losses['Hout'] = 'categorical_crossentropy'
                my_loss_weights["Hout"] = self.cath_loss_weight
                my_metrics["Hout"] = 'accuracy'
            if 6 in self.label_columns: 
                my_outputs.append("S35out")
                my_losses['S35out'] = 'categorical_crossentropy'
                my_loss_weights["S35out"] = self.cath_loss_weight
                my_metrics["S35out"] = 'accuracy'
            if 11 in self.label_columns: 
                #my_outputs.append("LENout")
                my_losses['LENout'] = self.domlength_regression_loss
                my_loss_weights["LENout"] = self.cath_domlength_weight
                #my_metrics["LENout"] = 'loss'
        
        return my_outputs, my_losses, my_loss_weights, my_metrics
    
    def get_cath_output_data(self): 
        output_dict = collections.OrderedDict()
        my_losses = {}#collections.OrderedDict()
        my_loss_weights = {}#collections.OrderedDict()
        my_metrics = {}#collections.OrderedDict()
        # attach four new output layers instead of removed output layer

        if not self.no_classifier:
            for l in self.label_columns:
                
                if l in ['cath01_class', "cath02_architecture", "cath03_topology", "cath04_homologous_superfamily",
                        "cath05_S35_cluster", "cath06_S60_cluster", "cath07_S95_cluster","cath08_S100_cluster"]:
                    print("adding loss for ",l)
                    output_dict[l] = pd.get_dummies( self.Ydict[l] )
                    my_losses[l] = 'categorical_crossentropy'
                    my_loss_weights[l] = self.cath_loss_weight
                    my_metrics[l] = 'accuracy'
            
                elif l in ["sequence_length"]: 
                    output_dict[l] = self.Ydict[l]
                    my_losses[l] = self.domlength_regression_loss
                    my_loss_weights[l] = self.cath_domlength_weight
                         
        if self.use_img_decoder:
            if self.model in ["seq2img_gan_gen"]:
                l = 'model_4'
            else:
                l = 'img_decoder'
            my_losses[l] = self.dc_decoder_loss
            output_dict[l] = self.X
            my_losses[l] = self.dc_decoder_loss
            my_loss_weights[l] = self.dc_dec_weight
            
        if self.use_seq_decoder:
            my_losses['seq_decoder'] = self.seq_decoder_loss
            my_losses["seq_decoder"] = self.seq_decoder_loss
            my_loss_weights["seq_decoder"] = self.seq_dec_weight  
            output_dict["seq_decoder"] = self.seqdata 
        
        return output_dict, my_losses, my_loss_weights, my_metrics
    
    def get_callbacks(self, orig_model=None):
        callbacks = []
        
        if self.reduce_lr:
            rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            callbacks.append(rlrop)
        
        if not self.early_stopping in ["none", "None"]:
            ea = EarlyStopping(monitor=self.early_stopping, min_delta=0.001, patience=20, verbose=1, mode='auto')
            callbacks.append(ea)
        
        if K.backend() == "tensorflow" and self.tensorboard:
            if os.path.exists(self.tbdir):
                tb = TensorBoard(log_dir=self.tbdir, histogram_freq=10, write_graph=True, write_images=True, write_grads=True)
                # https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter
                callbacks.append(tb)
        if self.checkpoints>0:
            chk = ModelCheckpointMultiGPU( os.path.join(self.outdir, self.modlabel+"_checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5"),
                                           orig_model, 
                                            monitor='val_loss', verbose=0, 
                                            save_best_only=True, save_weights_only=False, 
                                            mode='auto', period=self.checkpoints)
            callbacks.append(chk)
        
        csvlog = CSVLogger(os.path.join(self.outdir, self.modlabel+"_csvlogger.csv"), separator=',', append=False)
        callbacks.append(csvlog)
        time_callback = TimeHistory()
        callbacks.append(time_callback)
        return callbacks
    
    def get_optimizer(self):
        opt = None
        if self.optimizer == "sgd":
            if self.learning_rate == -1:
                opt = optimizers.SGD(momentum=0.01, decay=0.01, nesterov=False)
            else:
                opt = optimizers.SGD(lr=self.learning_rate, momentum=0.01, decay=0.01, nesterov=False)
        elif self.optimizer == "rmsprop":
            if self.learning_rate == -1:
                opt = optimizers.RMSprop()
            else:
                opt = optimizers.RMSprop(lr=self.learning_rate)
        elif self.optimizer == "adam":
            if self.learning_rate == -1:
                self.learning_rate = 0.001
                opt = optimizers.Adam(0.001, 0.9, 0.999, 1e-8, 0)  # defaults
            else:
                opt = optimizers.Adam(lr=self.learning_rate)
        else:
            print("ERROR: no valid optimizer:",opt)
            
        print ("Learning rate for optimizer %s: " % (self.optimizer))
        print (self.learning_rate)

        return opt

    def get_train_val_dicts(self, input_dict, output_dict):
        n_inputs = len(input_dict)
        n_outputs = len(output_dict)
        print(n_inputs, "inputs")
        print(n_outputs,"outputs")

        
        split = train_test_split( *([input_dict[i] for i in input_dict.keys()] + [output_dict[o] for o in output_dict.keys()])  ) 
        all_keys = [ ]
        for i in list(input_dict.keys()) + list(output_dict.keys()):
            all_keys.append(i)
            all_keys.append(i)
        
        train_input_dict=collections.OrderedDict()
        val_input_dict=collections.OrderedDict()
        train_output_dict=collections.OrderedDict()
        val_output_dict=collections.OrderedDict()

        for i in range(0, len(split),2):
            key = all_keys[i]
            if i < n_inputs:
                train_input_dict[key] = split[i]    
                val_input_dict[key] = split[i+1]
            else:
                train_output_dict[key] = split[i]
                val_output_dict[key] = split[i+1]

        
       
        print(train_input_dict.keys())
        print(val_input_dict.keys())

        print(train_output_dict.keys())
        print(val_output_dict.keys())
        if False:
            if not self.use_img_decoder:
                if self.use_seq_encoder and self.use_img_encoder:
                    split = train_test_split(input_dict["main_input"], input_dict["seq_input"], output_dict["cath01_class"], output_dict["cath02_architecture"], output_dict["cath03_topology"], output_dict["cath04_homologous_superfamily"], output_dict["sequence_length"], test_size = self.valsplit)
                    train_input_dict={"main_input":split[0], "seq_input":split[2]}
                    val_input_dict={"main_input":split[1], "seq_input":split[3]}

                    train_output_dict={"cath01_class":split[4], "cath02_architecture":split[6], "cath03_topology":split[8], "cath04_homologous_superfamily":split[10], "sequence_length":split[12]}
                    val_output_dict={"cath01_class":split[5], "cath02_architecture":split[7], "cath03_topology":split[9], "cath04_homologous_superfamily":split[11], "sequence_length":split[13]}
                elif self.use_seq_encoder and not self.use_img_encoder:
                    split = train_test_split(input_dict["seq_input"], output_dict["cath01_class"], output_dict["cath02_architecture"], output_dict["cath03_topology"], output_dict["cath04_homologous_superfamily"], output_dict["sequence_length"], test_size = self.valsplit)
                    train_input_dict={"seq_input":split[0]}
                    val_input_dict={"seq_input":split[1]}

                    train_output_dict={"cath01_class":split[2], "cath02_architecture":split[4], "cath03_topology":split[6], "cath04_homologous_superfamily":split[8], "sequence_length":split[10]}
                    val_output_dict={"cath01_class":split[3], "cath02_architecture":split[5], "cath03_topology":split[7], "cath04_homologous_superfamily":split[9], "sequence_length":split[11]}
                elif not self.use_seq_encoder and self.use_img_encoder:
                
                    split = train_test_split(input_dict["main_input"], output_dict["cath01_class"], output_dict["cath02_architecture"], output_dict["cath03_topology"], output_dict["cath04_homologous_superfamily"], output_dict["sequence_length"], test_size = self.valsplit)
                    train_input_dict={"main_input":split[0]}
                    val_input_dict={"main_input":split[1]}

                    train_output_dict={"cath01_class":split[2], "cath02_architecture":split[4], "cath03_topology":split[6], "cath04_homologous_superfamily":split[8], "sequence_length":split[10]}
                    val_output_dict={"cath01_class":split[3], "cath02_architecture":split[5], "cath03_topology":split[7], "cath04_homologous_superfamily":split[9], "sequence_length":split[11]}
            else:
                if self.no_classifier:
                    if self.model =="fullconv":
                        split = train_test_split(input_dict["main_input"],  output_dict["img_decoder"], test_size = self.valsplit)
                        train_input_dict={"main_input":split[0]}
                        val_input_dict={"main_input":split[1]}

                        train_output_dict={"img_decoder":split[2]}
                        val_output_dict={"img_decoder":split[3]}
                    elif self.model =="seq2img":
                        split = train_test_split(input_dict["seq_input"],  output_dict["img_decoder"], test_size = self.valsplit)
                        train_input_dict={"seq_input":split[0]}
                        val_input_dict={"seq_input":split[1]}

                        train_output_dict={"img_decoder":split[2]}
                        val_output_dict={"img_decoder":split[3]}
                else:
                    if self.use_seq_encoder and self.use_img_encoder:
                        split = train_test_split(input_dict["main_input"], input_dict["seq_input"], output_dict["cath01_class"], output_dict["cath02_architecture"], output_dict["cath03_topology"], output_dict["cath04_homologous_superfamily"], output_dict["sequence_length"], output_dict["img_decoder"], test_size = self.valsplit)
                        train_input_dict={"main_input":split[0], "seq_input":split[2]}
                        val_input_dict={"main_input":split[1], "seq_input":split[3]}

                        train_output_dict={"cath01_class":split[4], "cath02_architecture":split[6], "cath03_topology":split[8], "cath04_homologous_superfamily":split[10], "sequence_length":split[12], "img_decoder":split[14]}
                        val_output_dict={"cath01_class":split[5], "cath02_architecture":split[7], "cath03_topology":split[9], "cath04_homologous_superfamily":split[11], "sequence_length":split[13], "img_decoder":split[15]}
                    elif self.use_seq_encoder and not self.use_img_encoder and self.no_classifier:
                        split = train_test_split(input_dict["seq_input"],output_dict["img_decoder"], test_size = self.valsplit)
                        train_input_dict={"seq_input":split[0]}
                        val_input_dict={"seq_input":split[1]}

                        train_output_dict={"img_decoder":split[2]}
                        val_output_dict={"img_decoder":split[3]}
                    elif self.use_seq_encoder and not self.use_img_encoder and not self.no_classifier:
                        split = train_test_split(input_dict["seq_input"], output_dict["cath01_class"], output_dict["cath02_architecture"], output_dict["cath03_topology"], output_dict["cath04_homologous_superfamily"], output_dict["sequence_length"], output_dict["img_decoder"], test_size = self.valsplit)
                        train_input_dict={"seq_input":split[0]}
                        val_input_dict={"seq_input":split[1]}

                        train_output_dict={"cath01_class":split[2], "cath02_architecture":split[4], "cath03_topology":split[6], "cath04_homologous_superfamily":split[8], "sequence_length":split[10], "img_decoder":split[12]}
                        val_output_dict={"cath01_class":split[3], "cath02_architecture":split[5], "cath03_topology":split[7], "cath04_homologous_superfamily":split[9], "sequence_length":split[11], "img_decoder":split[13]}

        return train_input_dict, val_input_dict, train_output_dict, val_output_dict
    
    def kfold_train_cath(self, n_splits=5, shuffle=False, mode="StratifiedKFold", random_state=None, target_columns=[], group_column=None):
        
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
        for train_index, val_index in splitter.split(range(len(self.labels)),self.Ydict[target_columns], groups=groups):
            df_train_indices = self.act_df.iloc[train_index,].index
            df_val_indices = self.act_df.iloc[val_index,].index

            self.train_cath(generic_label="F%i_"%count, train_indices=df_train_indices, val_indices=df_val_indices)
            
            self.k_histories.append(self.history)
            self.k_modlabels.append(self.modlabel)
            count+=1
        
        for i in range(n_splits):
            h = self.k_histories[i]
            m = self.k_modlabels[i]
            print("Fold %i >>> loss: %d.4; val_loss: %d.4 --- %s"%(i+1, h.history["loss"][-1], h.history["val_loss"][-1],  m  ) )
    
    def train(self, generic_label="test", load_model=False, show_classnames=False): # deprecated
        if self.model == "none":
            print ("No model selected. Aborting.")
            return
        
        # get dataset
        self.prepare_dataset(show_classnames=show_classnames)
        
        if self.img_dim!=None: print ("Data set shape: %s" % str(self.X.shape))
        
        print ("CNN:")
      
        if self.img_dim!=None: print (self.X.shape)
        print ("Training this model:")
        # https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
        if self.ngpus == 1:
            self.keras_model = self.get_model()
        else:
            import tensorflow as tf

            with tf.device("/cpu:0"):
                self.orig_model = self.get_model()
                print ("Multi-GPU run - model on CPU:")
                print (self.orig_model)
                print(self.orig_model.summary())
                
            self.keras_model = multi_gpu_model(self.orig_model, gpus=self.ngpus)
        
        print (self.keras_model)
        print (self.keras_model.summary())
        
        # Let's train the model using RMSprop
        if self.optimizer == "sgd":
            if self.learning_rate == -1:
                opt = optimizers.SGD(momentum=0.01, decay=0.01, nesterov=False)
            else:
                opt = optimizers.SGD(lr=self.learning_rate, momentum=0.01, decay=0.01, nesterov=False)
        elif self.optimizer == "rmsprop":
            if self.learning_rate == -1:
                opt = optimizers.RMSprop()
            else:
                opt = optimizers.RMSprop(lr=self.learning_rate)
        elif self.optimizer == "adam":
            if self.learning_rate == -1:
                self.learning_rate = 0.001
                opt = optimizers.Adam(0.001, 0.9, 0.999, 1e-8, 0)  # defaults
            else:
                opt = optimizers.Adam(lr=self.learning_rate)
        else:
            print("ERROR: no valid optimizer")
            opt = None
        print ("Learning rate for optimizer %s: " % (self.optimizer))
        print (self.learning_rate)

        print (model.get_config())
        
        

        input_dict = collections.OrderedDict()

        if self.use_img_encoder:
            print("Using 2D input data", self.use_img_encoder, self.model)
            input_dict["main_input"] = self.X
        if self.use_seq_encoder:
            print("Using 1D input data")
            input_dict["seq_input"] = self.seqdata

        output_dict, my_losses, my_loss_weights, my_metrics = self.get_output_labels()
        
        # for classification outputs:
        output_dict = {my_outputs[o]:self.Ymats[o] for o in range(len(my_outputs))}
        
        # for regression outputs:
        if 11 in self.label_columns and not self.no_classifier:
            output_dict["LENout"] = self.Ys[-2]
        
                         
        if self.use_img_decoder:
            my_outputs.append("img_decoder")
            my_losses['img_decoder'] = self.dc_decoder_loss
            # output_labels = {my_outputs[o]:self.Ymats[o] for o in range(len(my_outputs)-1)}
            output_dict["img_decoder"] = self.X
            my_loss_weights["img_decoder"] = self.dc_dec_weight
            
        if self.use_seq_decoder:
            my_outputs.append("seq_decoder")
            my_losses['seq_decoder'] = self.seq_decoder_loss
            # output_dict = {my_outputs[o]:self.Ymats[o] for o in range(len(my_outputs)-1)}
            my_loss_weights["seq_decoder"] = self.seq_dec_weight  
            output_dict ["seq_decoder"] = self.seqdata 
    

        print (output_dict.keys())
        self.keras_model.compile(
                                 loss=my_losses,
                                 optimizer=opt,
                                 metrics=my_metrics,
                                 loss_weights=my_loss_weights
                                 )
 
        start = time.time()
        
        if False:#self.img_dim==None:
            # datagen = VarSizeImageDataGenerator()

            # # compute quantities required for featurewise normalization
            # # (std, mean, and principal components if ZCA whitening is applied)
            # datagen.fit(x_train)

            # # fits the model on batches with real-time data augmentation:
            # model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
            #                     steps_per_epoch=len(x_train) / 32, epochs=epochs)
            pass
        else:
            self.history = self.keras_model.fit(input_dict, output_dict,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.valsplit,
                shuffle=True,
                verbose=self.train_verbose
                , callbacks=self.get_callbacks()
                    )
        
        print (len(self.history.history["loss"]), "epochs trained")
        for k in self.history.history.keys():
            print ("%s:\t%.3f" % (k, self.history.history[k][-1]))
        
        
        end = time.time()
        elapsed = end - start
        print ("Elapsed training time (s): %s"%str(elapsed))

        self.modlabel += "_split%.2f_lo%.3f_vlo%.3f" % (self.valsplit, self.history.history["loss"][-1], self.history.history["val_loss"][-1])
        # print self.history.history
        
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
            self.orig_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(self.orig_model.to_json())
            
        return self.modlabel
        
    def train_cath(self, generic_label="test", load_model=False, show_classnames=False):
        
        # https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
        if self.ngpus == 1:
            self.keras_model = self.get_model()
        else:
            import tensorflow as tf

            with tf.device("/cpu:0"):
                self.orig_model = self.get_model()
                print ("Multi-GPU run - model on CPU:")
                print (self.orig_model)
                print(self.orig_model.summary())
                
            self.keras_model = multi_gpu_model(self.orig_model, gpus=self.ngpus)
        
        print (self.keras_model)
        print (self.keras_model.summary())
        #print (self.keras_model.get_config())
        
        
        opt = self.get_optimizer()


        input_dict = collections.OrderedDict()

        if self.use_img_encoder:
            print("Using 2D input data", self.use_img_encoder, self.model)
            input_dict["main_input"] = self.X
        if self.use_seq_encoder:
            print("Using 1D input data")
            input_dict["seq_input"] = self.seqdata


        output_dict, my_losses, my_loss_weights, my_metrics = self.get_cath_output_data()
        
        print(output_dict.keys())
        print(my_losses)
        print(my_loss_weights)
        print(my_metrics)
        
    

        print (output_dict.keys())
        self.keras_model.compile(
                                 loss=my_losses,
                                 optimizer=opt,
                                 metrics=my_metrics,
                                 loss_weights=my_loss_weights
                                 )
 
        start = time.time()
        
        #print(input_dict)

        #print(output_dict)
        train_input_dict, val_input_dict, train_output_dict, val_output_dict = self.get_train_val_dicts(input_dict, output_dict)

        self.history = self.keras_model.fit(
                                            train_input_dict, 
                                            train_output_dict,
                                            batch_size=self.batch_size,
                                            epochs=self.epochs,
                                            validation_data=(val_input_dict, val_output_dict),
                                            shuffle=True,
                                            verbose=self.train_verbose,
                                            callbacks=self.get_callbacks(orig_model=self.orig_model if self.ngpus >  1 else self.keras_model)
                                        )
        
        print (len(self.history.history["loss"]), "epochs trained")
        for k in self.history.history.keys():
            print ("%s:\t%.3f" % (k, self.history.history[k][-1]))
        
        
        end = time.time()
        elapsed = end - start
        print ("Elapsed training time (s): %s"%str(elapsed))

        self.modlabel += "_split%.2f_lo%.3f_vlo%.3f" % (self.valsplit, self.history.history["loss"][-1], self.history.history["val_loss"][-1])
        # print self.history.history
        
        self.model_arch_file = os.path.join(self.outdir, "%s_model_arch.json" % self.modlabel)
        self.model_weights_file = os.path.join(self.outdir, "%s_model_weights.hdf5" % self.modlabel)
        
        if self.ngpus == 1:
            self.save_training_history()
            #with open(os.path.join(self.outdir, "%s_model_history.json" % self.modlabel), "w") as hist:
            #    hist.write(str(self.history.history))
            self.keras_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(self.keras_model.to_json())
            plot_model(self.keras_model, to_file=os.path.join(self.outdir, self.modlabel+'_model-diagram.png'),show_shapes=True, show_layer_names=True, rankdir='TB')#'LR'

        else:
            self.save_training_history()
            #with open(os.path.join(self.outdir, "%s_model_history.json" % self.modlabel), "w") as hist:
            #    hist.write(str(self.history.history))
            self.orig_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(self.orig_model.to_json())
            
            plot_model(self.orig_model, to_file=os.path.join(self.outdir, self.modlabel+'_model-diagram.png'),show_shapes=True, show_layer_names=True, rankdir='TB')#'LR'

        return self.modlabel   
    
    def generator_train_h5(self, generic_label="test", load_model=False, show_classnames=False, train_indices=None, val_indices=None):
        if self.model == "none":
            print ("No model selected. Aborting.")
            return
        
        # get dataset
        self.prepare_dataset(show_classnames=show_classnames)
        
        if self.img_dim!=None: print ("Data set shape: %s" % str(self.X.shape))
        
        print ("CNN:")
      
        if self.img_dim!=None: print (self.X.shape)
        print ("Training this model:")
        # https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
        if self.ngpus == 1:
            self.keras_model = self.get_model()
        else:
            import tensorflow as tf

            with tf.device("/cpu:0"):
                self.orig_model = self.get_model()
                print ("Multi-GPU run - model on CPU:")
                print (self.orig_model)
                print(self.orig_model.summary())
                
            self.keras_model = multi_gpu_model(self.orig_model, gpus=self.ngpus)
        
        print (self.keras_model)
        print (self.keras_model.summary())
        
        # Let's train the model using RMSprop
        if self.optimizer == "sgd":
            if self.learning_rate == -1:
                opt = optimizers.SGD(momentum=0.01, decay=0.01, nesterov=False)
            else:
                opt = optimizers.SGD(lr=self.learning_rate, momentum=0.01, decay=0.01, nesterov=False)
        elif self.optimizer == "rmsprop":
            if self.learning_rate == -1:
                opt = optimizers.RMSprop()
            else:
                opt = optimizers.RMSprop(lr=self.learning_rate)
        elif self.optimizer == "adam":
            if self.learning_rate == -1:
                self.learning_rate = 0.001
                opt = optimizers.Adam(0.001, 0.9, 0.999, 1e-8, 0)  # defaults
            else:
                opt = optimizers.Adam(lr=self.learning_rate)
        else:
            print("ERROR: no valid optimizer")
            opt = None
        print ("Learning rate for optimizer %s: " % (self.optimizer))
        print (self.learning_rate)
        # X_train = X_train.astype('float32')
        # X_test = X_test.astype('float32')
        # X_train /= 255
        # X_test /= 255

        # print model.get_config()
        
        callbacks = []
        
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        callbacks.append(rlrop)
        
        if not self.early_stopping in ["none", "None"]:
            ea = EarlyStopping(monitor=self.early_stopping, min_delta=0.001, patience=20, verbose=1, mode='auto')
            callbacks.append(ea)
        
        if K.backend() == "tensorflow" and self.tensorboard:
            if os.path.exists(self.tbdir):
                tb = TensorBoard(log_dir=self.tbdir, histogram_freq=10, write_graph=True, write_images=True, write_grads=True)
                # https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter
                callbacks.append(tb)
        if self.checkpoints>0:
            chk = ModelCheckpointMultiGPU( os.path.join(self.outdir, self.modlabel+"_checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5"),
                                           self.orig_model, 
                                            monitor='val_loss', verbose=0, 
                                            save_best_only=True, save_weights_only=False, 
                                            mode='auto', period=self.checkpoints)
            callbacks.append(chk)
        
        time_callback = TimeHistory()
        callbacks.append(time_callback)

        my_input_data = {}

        if self.use_img_encoder:
            print("Using 2D input data", self.use_img_encoder, self.model)
            my_input_data["main_input"] = self.X
        
        my_outputs, my_losses, my_loss_weights, my_metrics = self.get_output_labels()
        
        # for classification outputs:
        output_labels = {my_outputs[o]:self.Ymats[o] for o in range(len(my_outputs))}
        
        # for regression outputs:
        if 11 in self.label_columns and not self.no_classifier:
            output_labels["LENout"] = self.Ys[-2]
        
                         
        if self.use_img_decoder:
            my_outputs.append("dc_decoder")
            my_losses['dc_decoder'] = self.dc_decoder_loss
            # output_labels = {my_outputs[o]:self.Ymats[o] for o in range(len(my_outputs)-1)}
            output_labels["dc_decoder"] = self.X
            my_loss_weights["dc_decoder"] = self.dc_dec_weight
            
        if self.use_seq_decoder:
            my_outputs.append("seq_dec")
            my_losses['seq_dec'] = self.seq_decoder_loss
            # output_labels = {my_outputs[o]:self.Ymats[o] for o in range(len(my_outputs)-1)}
            my_loss_weights["seq_dec"] = self.seq_dec_weight  
            output_labels ["seq_dec"] = self.seqdata 
        if self.use_seq_encoder:
            print("Using 1D input data")
            my_input_data["seq_input"] = self.seqdata   

        print (output_labels.keys())
        self.keras_model.compile(
                                 loss=my_losses,
                                 optimizer=opt,
                                 metrics=my_metrics,
                                 loss_weights=my_loss_weights
                                 )
        

        

        start = time.time()

        if self.img_bin_batch_sizes==[]:
            if False:#self.img_dim==None:
                # datagen = VarSizeImageDataGenerator()

                # # compute quantities required for featurewise normalization
                # # (std, mean, and principal components if ZCA whitening is applied)
                # datagen.fit(x_train)

                # # fits the model on batches with real-time data augmentation:
                # model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                #                     steps_per_epoch=len(x_train) / 32, epochs=epochs)
                pass
            else:
                if train_indices==None or val_indices==None:
                    all_indices = range(self.X.shape[0])
                    random.shuffle(all_indices)

                    train_indices = all_indices[:int(math.ceil(len(all_indices)*(1.0-self.valsplit) ))]#  [int(i) for i in range(self.X.shape[0]) if i%2==0 ] 
                    #random.shuffle(train_indices)

                    val_indices = all_indices[int(math.ceil(len(all_indices)*(1.0-self.valsplit) )) : ]#[int(i) for i in range(self.X.shape[0]) if i%2!=0 ]
                    #random.shuffle( val_indices )
                
                
                training_generator   = H5_Image_Generator(
                                                        indices=train_indices, 
                                                        inputs=my_input_data, 
                                                        outputs=output_labels, 
                                                        batch_size=self.batch_size,
                                                        is_train=True,
                                                        crops_per_image=self.crops_per_image, 
                                                        crop_width=self.crop_width, 
                                                        crop_height=self.crop_height
                                                        )
                validation_generator = H5_Image_Generator(
                                                        indices=val_indices, 
                                                        inputs=my_input_data, 
                                                        outputs=output_labels, 
                                                        batch_size=self.batch_size,
                                                        is_train=False,
                                                        crops_per_image=self.crops_per_image, 
                                                        crop_width=self.crop_width, 
                                                        crop_height=self.crop_height
                                                        )    
                
                self.history = self.keras_model.fit_generator(
                        generator=training_generator,
                        steps_per_epoch=max(1,self.crops_per_image) * math.ceil(len(train_indices) / self.batch_size) ,
                        epochs=self.epochs,
                        verbose=self.train_verbose,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=max(1,self.crops_per_image) * math.ceil(len(val_indices)/ self.batch_size),
                        max_queue_size=20,
                        workers=24,
                        use_multiprocessing=True,
                        shuffle=True 
                        )
            print (len(self.history.history["loss"]), "epochs trained")
            for k in self.history.history.keys():
                print ("%s:\t%.3f" % (k, self.history.history[k][-1]))
        else:
            histories = []
            
            # train on image sizes with most data, then ascend to largest data set
            s_largest_set_last, _, s_batch_sizes = zip(*sorted(zip(self.img_size_bins,
                                                                    self.sn,
                                                                    self.img_bin_batch_sizes), reverse=False, key=lambda x:x[1])) 
            print("img sizes:", s_largest_set_last)
            print("batch sizes:", s_batch_sizes)
            temp_weights = self.keras_model.get_weights()
            for si in range(len(s_largest_set_last)):
                s = s_largest_set_last[si]
                output_labels = {my_outputs[o]:self.dYmats[s][o] for o in range(len(my_outputs))}
                print (s)
                for lab in output_labels:
                    np.transpose(output_labels[lab])
                    print (lab, output_labels[lab].shape)
                
                print(self.dX[s].shape)
                
                self.keras_model.set_weights(temp_weights)
                self.history = self.keras_model.fit(self.dX[s], output_labels,
                  batch_size=s_batch_sizes[si],
                  epochs=self.epochs,
                  validation_split=self.valsplit,
                  shuffle=True,
                  verbose=self.train_verbose
                  , callbacks=callbacks
                    )
                temp_weights = self.keras_model.get_weights()
                histories.append(self.history)
                print (len(self.history.history["loss"]), "epochs trained")
                for k in self.history.history.keys():
                    print ("%s:\t%.3f" % (k, self.history.history[k][-1]))
            self.history = histories[0]
            for h in histories[1:]:
                for key in self.history:
                    self.history[key].append(h[key])
        
        end = time.time()
        elapsed = end - start
        print ("Elapsed training time (s): %s"%str(elapsed))

        self.modlabel += "_split%.2f_lo%.3f_vlo%.3f" % (self.valsplit, self.history.history["loss"][-1], self.history.history["val_loss"][-1])
        # print self.history.history
        
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
            self.orig_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(self.orig_model.to_json())
            
        return self.modlabel
    
    def plot_curves(self, metrics=["loss"]):
        plt.rcParams['figure.figsize']=(20,20)
        if self.history == None: 
            print("No history found.")
            return
        #cath_label_map = {"Cout":"C", "Aout":"A", "Tout":"T", "Hout":"H"}
        
        # rf_metric = "acc"#"f1"#"f1"
        #train_metrics = metrics  # "fbeta_score"#"fbeta_score"
        # test_metric = "val_"+train_metric
        #color_map = {"Cout":"blue", "Aout":"green", "Tout":"red", "Hout":"cyan"}
        color_list=["blue","green","red","cyan", "orange","magenta", "black"]
        # fig = plt.subplots(2,2)
        
        all_train = {}
        all_test = {}
        d = self.history.history
        print (d.keys())
        rf_val = None
            
        actual_ep = len(d["loss"])
        print ("found %i epochs" % (actual_ep))
        
        counter=0
        for train_metric in metrics:
            if not train_metric in d: continue
                
            train_label = train_metric
            test_label = "val_" + train_metric
            if not test_label in d.keys():
                test_label = ""
            
            print(train_label, test_label)
            col = color_list[counter]
            #if train_metric == "loss":
            #    print("min(%s) for training (validation) '%s' = %.3f (%.3f)" % (train_label, cath_label_map[out], min(d[train_label]), min(d[test_label])))
            #else:
            #    print("max(%s) for training (validation) '%s' = %.3f (%.3f)" % (train_label, cath_label_map[out], max(d[train_label]), max(d[test_label])))
            
            plt.plot(d[train_label], linestyle="-", color=col, label=train_metric)
            if test_label!="": plt.plot(d[test_label], linestyle=":", color=col, label=None)
            
            if d[train_label][-1] != 0: plt.annotate(str(round_figures(d[train_label][-1], 3)) + "(train)", (actual_ep, d[train_label][-1]), xycoords='data', color=col)
            if test_label!="": 
                if d[test_label][-1] != 0: 
                    plt.annotate(str(round_figures(d[test_label][-1], 3)) + "(test)", (actual_ep, d[test_label][-1]), xycoords='data', color=col)
            
            all_train[train_metric] = d[train_metric]
            if test_label!="": all_test[train_metric] = d["val_"+ train_metric]
            counter += 1
        
        if rf_val != None: 
            plt.plot([rf_val for i in range(self.epochs)], linestyle="--", color="black", label=None)
            plt.annotate(str(round_figures(rf_val, 3)) + "(RF)", (actual_ep, rf_val), xycoords='data', color="black")
        plt.xlabel("epoch")
        plt.ylabel("metric:" + train_metric)
        plt.title(self.modlabel)
        # if test_label!="": 
        #     max_sum = None  
        #     max_sum_t = None             
        #     for t in range(actual_ep):
        #         s_train = 0
        #         s_test = 0
        #         for out in ["Cout", "Aout", "Tout", "Hout"]:
        #             if out == "Cout" and not 2 in self.label_columns: continue
        #             if out == "Aout" and not 3 in self.label_columns: continue
        #             if out == "Tout" and not 4 in self.label_columns: continue
        #             if out == "Hout" and not 5 in self.label_columns: continue
        #             s_train += all_train[out][t]
        #             s_test += all_test[out][t]
        #         s = s_train + s_test
        #         if s > max_sum:
        #             max_sum = s
        #             max_sum_t = t
            
        #     print("Best t = %i (score=%.3f)" % (max_sum_t, max_sum / 8))
            
        if train_metric == "loss":
            plt.legend(loc="upper right")
        else:
            plt.legend(loc="lower right")
        
        # pngname = str(fullpath).replace(".json","_%s_plot.png"%(train_metric))
        # print( pngname )
        #plt.tight_layout()
        
        save_file = os.path.join(os.path.abspath(self.outdir), self.modlabel+"_"+metrics[0]+"_training-curve.png")
        print("Saving plot:",save_file)
        plt.savefig( save_file )
        plt.close('all')
        
    def plot_sample_images(self, indices=[]):
        
        # show some random example images
        fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, squeeze=False)
        if len(indices)==0:
           indices = [random.randint(0, self.X.shape[0] - 1) for i in range(4)]
        else:
            indices = indices[:4]
        test_imgs = [np.squeeze(self.X[i]) if self.nchannels == 1 else self.X[i] for i in indices ]
        titles = [self.labels[i] for i in indices]
        # test_img = X[0,:,:,:]
        print ("random test images:", test_imgs[0].shape)
        # if self.nchannels==1: 
        #    test_img=np.squeeze(test_img)
        #    print ("test image squeezed:",test_img.shape)
        # test_img *= test_img*255
        axes[0, 0].imshow(test_imgs[0],  interpolation="none");axes[0, 0].set_title(titles[0])
        axes[0, 1].imshow(test_imgs[1],  interpolation="none");axes[0, 1].set_title(titles[1])
        axes[1, 0].imshow(test_imgs[2],  interpolation="none");axes[1, 0].set_title(titles[2])
        axes[1, 1].imshow(test_imgs[3],  interpolation="none");axes[1, 1].set_title(titles[3])
        # le = LabelEncoder()
        plt.tight_layout()
        #plt.show()
        print("writing: ",os.path.join(self.outdir, self.modlabel+"_random_images.png"))
        plt.savefig(os.path.join(self.outdir, self.modlabel+"_random_images.png"))
        plt.close('all')
        
    def visualize_image_data(self, show_plots=False, use_pixels=False, use_pfp=True, pfp_layername="fc_last", ndims=2, perp=100
        , early_exaggeration=4.0 
        , learning_rate=100, angle=0.5
        , n_iter=1000, rseed=123, pfp_length=512, marker_size=1, n_jobs=1, save_path="", do_cluster_analysis=False, clustering_method="kmeans", scores_dict={}): # deprecated
        
        
        assert clustering_method in ["kmeans", "hdbscan"]

        files_written = []
        
        if use_pfp:
            if self.verbose: print("Generating protein fingerprints...")
            from ProtConv2D.predictChembl import get_pfp_generator_model
            pfp_generator = get_pfp_generator_model(self.model_arch_file, self.model_weights_file, pfp_layername)
            if self.use_seq_encoder:
                X_pfp = pfp_generator.predict(self.seqdata)
            else:
                X_pfp = pfp_generator.predict(self.X)
        
        
        method = "barnes_hut"; assert method in ["barnes_hut", "exact"]
        metric = 'cosine'; assert metric in ["braycurtis", "canberra", "chebyshev", "cityblock",
                                                          "correlation", "cosine", "dice", "euclidean", "hamming",
                                                          "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski",
                                                          "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener",
                                                          "sokalsneath", "sqeuclidean", "yule"]
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        names = ["Class", "Architecture", "Topology", "Homology"]
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
        
        if use_pixels:
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
            for l in self.Ys[:4]:
                if len(l) == 0:continue
                
                labels = le.fit_transform(l)
                
                if do_cluster_analysis:
                    if clustering_method=="hdbscan":
                        import hdbscan
                        clu = hdbscan.HDBSCAN(metric="euclidean",min_cluster_size=self.nb_classes[count])
                    elif clustering_method=="kmeans":
                        clu = KMeans(n_clusters=self.nb_classes[count])
                    clu.fit(X)
                    score = homogeneity_score(labels, clu.labels_)
                    scores_dict["pixels_%s_homogeneity_score"%names[count]] = score
                    print( names[count]," score:",score)
                    clusters.append(clu.labels_)
                else:
                    score = -1
                
                cmap = cm.get_cmap(name="gist_ncar")
                c = [cmap(float(i) / self.nb_classes[count]) for i in labels]
    
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
                ax.set_title("%s (%i classes)%s" % (names[count], self.nb_classes[count], " score=%s"%str(score) if score>=0 else ""))
                # ax.set_title(title)
                # if savename!=None: plt.savefig(savename,dpi=200)
                # plt.colorbar(cax)
                
                # plt.show()
                # plt.clf()
                count += 1
            if show_plots: plt.show()
            plt.savefig(os.path.join(save_path, self.modlabel+"_dataviz_pixels.png"))
            files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pixels.png"))
            df = pd.DataFrame({"tsne1":x1, "tsne2":x2, "Class":self.Ys[0], "Architecture":self.Ys[1], "Topology":self.Ys[2], "Homology":self.Ys[3]} 
                         ,columns=["Class","Architecture","Topology","Homology","tsne1","tsne2"], index=self.labels)
            if 11 in self.label_columns:
                domlen = self.Ys[-2]
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
            for l in self.Ys[:4]:
                if len(l) == 0:continue
                labels = le.fit_transform(l)
                
                if do_cluster_analysis:
                    if clustering_method=="hdbscan":
                        import hdbscan
                        clu = hdbscan.HDBSCAN(metric="euclidean",min_cluster_size=self.nb_classes[count])
                    elif clustering_method=="kmeans":
                        clu = KMeans(n_clusters=self.nb_classes[count])
                    clu.fit(X_pfp)
                    score = homogeneity_score(labels, clu.labels_)
                    scores_dict["pfprints_%s_homogeneity_score"%names[count]] = score
                    print( names[count]," score:",score)
                    clusters.append(clu.labels_)
                else:
                    score = -1
                
                cmap = cm.get_cmap(name="gist_ncar")
                c = [cmap(float(i) / self.nb_classes[count]) for i in labels]
                
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
                ax.set_title("%s (%i classes)%s" % (names[count], self.nb_classes[count], " score=%s"%str(score) if score>=0 else ""))               
                # if savename!=None: plt.savefig(savename,dpi=200)
                # plt.colorbar(cax)
                
                # plt.show()
                # plt.clf()
                count += 1
            
            
            if show_plots: plt.show()
            plt.savefig(os.path.join(save_path, self.modlabel+"_dataviz_pfprints.png"))
            files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pfprints.png"))
            df = pd.DataFrame({"tsne1":x1, "tsne2":x2, "Class":self.Ys[0], "Architecture":self.Ys[1], "Topology":self.Ys[2], "Homology":self.Ys[3]} 
                         ,columns=["Class","Architecture","Topology","Homology","tsne1","tsne2"], index=self.labels)
            
            if 11 in self.label_columns:
                print("domain length info available.")
                domlen = self.Ys[-2]
                plt.scatter(x1, x2, s=marker_size, c=domlen, lw=.5)  # ,cmap=colmap)
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.colorbar()
                plt.title("input: protein fingerprints of length %i\nt-SNE with perplexity %.3f\ndomain length" % (self.fc2, perp))               
                plt.savefig(os.path.join(save_path, self.modlabel+"_dataviz_pfprints_domain_length.png"))
                files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pfprints_domain_length.png"))
            
            if len(clusters)!=0:
                df["C_clusters"] = clusters[0]
                df["A_clusters"] = clusters[1]
                df["T_clusters"] = clusters[2]
                df["H_clusters"] = clusters[3]
            df.to_csv(os.path.join(save_path, self.modlabel+"_dataviz_pfprints.csv"), index_label="cath_id" )
            files_written.append(os.path.join(save_path, self.modlabel+"_dataviz_pfprints.csv"))
        plt.close('all')
        return files_written

    def get_pfp_generator_model(self, model_arch_file, model_weights_file, output_layer):
        with open(model_arch_file) as archfile:
            model = model_from_json(archfile.read())
        model.load_weights(model_weights_file)

        if not type(output_layer)==list:
            output_layer = [output_layer]
        my_layer =None
        for layer_name in output_layer:
            if type(my_layer)==type(None):
                for layer in model.layers:
                    print(layer.name)
                    if layer.name == layer_name:
                        my_layer = layer
                        print("      found")
                    
        if type(my_layer)==type(None):
            print("ERROR: could not find this layer to create a PFP generator:",output_layer)
        if len(my_layer.output_shape)==5:
            return Model(inputs=model.input, outputs=GlobalAveragePooling3D()(my_layer.output))
        elif len(my_layer.output_shape)==4:
            return Model(inputs=model.input, outputs=GlobalAveragePooling2D()(my_layer.output))
        elif len(my_layer.output_shape)==3:
            return Model(inputs=model.input, outputs=GlobalAveragePooling1D()(my_layer.output))
        else:
            return Model(inputs=model.input, outputs=my_layer.output)
    
    def visualize_cath_image_data(self, show_plots=False, use_pixels=False, use_pfp=True, pfp_layername="fc_last", ndims=2, perp=100
        , early_exaggeration=4.0 
        , learning_rate=100, angle=0.5
        , n_iter=1000, rseed=123, marker_size=1, n_jobs=1, save_path="", do_cluster_analysis=False, clustering_method="kmeans", scores_dict={}): 
        
        plt.rcParams['figure.figsize']=(20,20)
        assert clustering_method in ["kmeans", "hdbscan"]

        files_written = []
        
        if use_pfp:
            
            pfp_generator = self.get_pfp_generator_model(self.model_arch_file, self.model_weights_file, pfp_layername)
            inputs=pfp_generator.inputs
            input_names = [i.name for i in inputs]

            print(input_names)
            if self.verbose: print("Generating protein fingerprints...")

            if len(input_names)==2:
                X_pfp = pfp_generator.predict([self.X, np.array(self.seqdata)])
            elif len(input_names)==1:
                if "seq_input"in input_names[0]:
                    X_pfp = pfp_generator.predict(np.array(self.seqdata))
                else:
                    X_pfp = pfp_generator.predict(self.X)
        print("PFP shape:",X_pfp.shape)
        pfp_length = X_pfp.shape[1]
        print("pfp_length",pfp_length)
        
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
            df = pd.DataFrame({"tsne1":x1, "tsne2":x2, "Class":self.Ydict["cath01_class"], "Architecture":self.Ydict["cath02_architecture"], "Topology":self.Ydict["cath03_topology"], "Homology":self.Ydict["cath04_homologous_superfamily"]} 
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

    def plot_reconstruction_samples_seq2img(self):
        plt.rcParams['figure.figsize']=(40,40)

        seq_dim = self.seqdata.shape[-1]
        layer_dict={l.name:l for l in self.keras_model.layers}
        #print(self.keras_model.layers)
        if self.ngpus==1:
            input2d_model = Model(inputs=self.keras_model.input, outputs=layer_dict["self_merge_lstm"].output)
        else:
            input2d_model = Model(inputs=self.orig_model.input, outputs=layer_dict["self_merge_lstm"].output)
        for i in range(0, 32,6):
            print (i, i+1)
            
            s=self.seqdata[i]

            print(s.shape, seq_dim)
            if len(s.shape)==1:
                s= s.reshape((1,self.manual_seq_maxlength,1))
                seq_dim = 1
            else:
                s= s.reshape((1,self.manual_seq_maxlength,seq_dim))
            print(s.shape)
            d=input2d_model.predict( s )
            print (d.shape)
            if d.shape[-1]>1 and d.shape[-1]<10:
                d = d.reshape((self.img_dim,self.img_dim,self.nchannels))
            else:
                d = d.reshape((self.img_dim,self.img_dim))
            print (d.shape)
            x = np.squeeze(self.X[i]).reshape((1,self.img_dim,self.img_dim,self.nchannels))
            print (x.shape)
            if self.no_classifier or len(self.keras_model.outputs)==1:
                print("single output found")
                dec = self.keras_model.predict(s.reshape((1,self.manual_seq_maxlength,seq_dim)))
            else:
                if "sequence_length" in self.label_columns:
                    C,A,T,H,L,dec = self.keras_model.predict(s.reshape((1,self.manual_seq_maxlength,seq_dim)))
                    print(C,A,T,H,L)
                else:
                    C,A,T,H,dec = self.keras_model.predict(s.reshape((1,self.manual_seq_maxlength,seq_dim)))
                    print(C,A,T,H)
            
            print( dec.reshape((self.img_dim,self.img_dim,self.nchannels)).shape )
            
            # processed input
            plt.subplot(8,6,1 + i)
            im_o=plt.imshow(np.log( d.astype('float') ),aspect='equal')
            plt.title("sequence input --> 2D (log)")
            plt.colorbar(im_o)
            plt.subplot(8,6,2 + i)
            sns.distplot(d.flatten(),bins=256,kde=False, color='r').set_yscale('log')
            #histo.set_yscale('log');
            plt.xlabel("normalized values");plt.ylabel("log counts");plt.title("matrix histogram: input");plt.show()
            #plt.colorbar(im)   
            
            # ground truth
            plt.subplot(8,6,3 + i)
            plt.imshow(x.astype('int').reshape(self.img_dim,self.img_dim,self.nchannels), aspect='equal')
            plt.title("ground truth")
            plt.colorbar()
            plt.subplot(8,6,4 + i)
            sns.distplot(x.astype("int").flatten(),bins=256,kde=False, color='r').set_yscale('log')
            #histo.set_yscale('log');
            plt.xlabel("normalized values");plt.ylabel("log counts");plt.title("matrix histogram: ground truth");plt.show()

            #reconstruction
            plt.subplot(8,6,5 + i)
            plt.imshow(dec.astype('int').reshape(self.img_dim,self.img_dim,self.nchannels), aspect='equal')
            plt.title("reconstruction")
            plt.colorbar()
            plt.subplot(8,6,6 + i)
            sns.distplot(dec.astype("int").flatten(),bins=256,kde=False, color='r').set_yscale('log')
            #histo.set_yscale('log');
            plt.xlabel("normalized values");plt.ylabel("log counts");plt.title("matrix histogram: reconstruction");plt.show()

            plt.tight_layout()

        plt.savefig(os.path.join(self.outdir, "%s_image_grid.png"%self.modlabel))
        plt.close('all')
    
    def plot_reconstruction_samples_unet(self):
        plt.rcParams['figure.figsize']=(40,40)
        
        for i in range(0, 32,4):
            print (i, i+1)
            

            x = np.squeeze(self.X[i]).reshape((1,self.img_dim,self.img_dim,self.nchannels))
            print (x.shape)
            
            dec = self.keras_model.predict(x)
            
            
            print( dec.reshape((self.img_dim,self.img_dim,self.nchannels)).shape )
            
            
            
            # ground truth
            plt.subplot(8,4,1 + i)
            plt.imshow(x.astype('int').reshape(self.img_dim,self.img_dim,self.nchannels), aspect='equal')
            plt.title("ground truth")
            plt.colorbar()
            plt.subplot(8,4,2 + i)
            sns.distplot(x.astype("int").flatten(),bins=256,kde=False, color='r').set_yscale('log')
            #histo.set_yscale('log');
            plt.xlabel("normalized values");plt.ylabel("log counts");plt.title("matrix histogram: ground truth");plt.show()

            #reconstruction
            plt.subplot(8,4,3 + i)
            plt.imshow(dec.astype('int').reshape(self.img_dim,self.img_dim,self.nchannels), aspect='equal')
            plt.title("reconstruction")
            plt.colorbar()
            plt.subplot(8,4,4 + i)
            sns.distplot(dec.astype("int").flatten(),bins=256,kde=False, color='r').set_yscale('log')
            #histo.set_yscale('log');
            plt.xlabel("normalized values");plt.ylabel("log counts");plt.title("matrix histogram: reconstruction");plt.show()

            plt.tight_layout()

        plt.savefig(os.path.join(self.outdir, "%s_image_grid.png"%self.modlabel))
        plt.close('all')
    
    def load_training_history(self, filename):
        hist_df = pd.read_csv(filename)
        self.history = History()
        self.history.history = hist_df.to_dict(orient='list')

    def save_training_history(self):
        results = pd.DataFrame(data=self.history.history)
        if self.verbose: print(results)
        results.to_csv( os.path.join(self.outdir, self.modlabel+"_results-history.csv"), index=False )

class H5_Image_Generator(Sequence):

    def __init__(self, indices, inputs, outputs, batch_size, is_train, shuffle=True, crops_per_image=0, crop_width=32, crop_height=32):
        self.nsamples = inputs[ inputs.keys()[0] ].shape[0]
        self.is_train = is_train

        print("n samples:",self.nsamples)
        self.indices = indices
        print ("H5_Image_Generator")
        print([inputs[i].shape for i in inputs], [outputs[o].shape for o in outputs])

        self.inputs = {i: np.take(a=inputs[i], indices=self.indices, axis=0) for i in inputs}
        self.outputs = {o: np.take(a=outputs[o], indices=self.indices, axis=0) for o in outputs}

        print([self.inputs[i].shape for i in self.inputs], [self.outputs[o].shape for o in self.outputs])
        self.batch_size = batch_size
        self.crops_per_image = crops_per_image
        self.effective_batch_size = self.batch_size if crops_per_image<=0 else self.batch_size/self.crops_per_image

        print("batch size ",self.batch_size )
        print("n crops ",self.crops_per_image)
        print("eff bs ",self.effective_batch_size)
        self.shuffle = shuffle
        
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.on_epoch_end()

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
        for inp in self.inputs:

            batches_x[inp] = self.inputs[inp][idx * self.effective_batch_size:(idx + 1) * self.effective_batch_size,   ]
            if self.crops_per_image>0:
                #print("batch",inp, batches_x[inp].shape)
                if inp=="main_input":
                    batch_crops=np.ndarray((self.batch_size, self.crop_width, self.crop_height, 3))
                    #print (batch_crops.shape)
                   
                    for i1 in range(self.effective_batch_size):
                        for i2 in range(self.crops_per_image):
                            img = batches_x[inp][i1,:,:,:]
                            #print("img ", img.shape)
                            crop = self.random_crop(img, (self.crop_width, self.crop_height))
                            #print("crop ",i1,i2,crop.shape)
                            batch_crops[i1+i2,:,:] = crop
                    #print("old batch:",batches_x[inp].shape)
                    #print("new batch:",batch_crops.shape)
                    batches_x[inp] = batch_crops
                else:
                    shape =  batches_x[inp].shape
                    shape = tuple([self.batch_size]+list(shape[1:]))

                    batch_tmp = np.ndarray( shape )
                    #print(inp, shape, batch_tmp.shape)
                    for i1 in range(self.effective_batch_size):
                        for i2 in range(self.crops_per_image):
                            batch_tmp[i1+i2,] = batches_x[inp][i1]
                    #print("old batch:",batches_x[inp].shape)
                    #print("new batch:",batch_tmp.shape)
                    batches_x[inp] = batch_tmp
            #print("input" ,inp, idx, batches_x[inp].shape, self.is_train)
            #assert len(batches_x[i]) > 0
        
        batches_y = {}
        for o in self.outputs:
            batches_y[o] = self.outputs[o][idx * self.effective_batch_size:(idx + 1) * self.effective_batch_size,   ]
            if self.crops_per_image>0:
                shape =  batches_y[o].shape
                shape = tuple([self.batch_size]+list(shape[1:]))

                batch_tmp = np.ndarray( shape )
                #print(o, shape, batch_tmp.shape)
                for i1 in range(self.effective_batch_size):
                    for i2 in range(self.crops_per_image):
                        batch_tmp[i1+i2,] = batches_y[o][i1]
                #print("old batch:",batches_y[o].shape)
                #print("new batch:",batch_tmp.shape)
                batches_y[o] = batch_tmp
            #print("output",o, idx, batches_y[o].shape, self.is_train)
            #assert len(batches_y[o]) > 0

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
    
