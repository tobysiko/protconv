# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:30:46 2017

@author: ts149092
"""

from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import ProtConv2D.MLhelpers as MLhelpers
from ProtConv2D import Conv_Seq_Dec_Softmax_Layer
#from PIL import Image
#import PIL.ImageOps  
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.activations import softmax
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
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, AveragePooling2D, MaxPooling1D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Conv1D, Conv2D, AveragePooling2D, AveragePooling1D, UpSampling1D, UpSampling2D, Reshape, merge
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.merge import add, multiply, dot, subtract, average, concatenate, maximum
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json, clone_model
from keras.preprocessing import sequence
from keras.regularizers import l1, l2
from keras.utils import layer_utils, np_utils, multi_gpu_model, plot_model
from keras.utils.data_utils import get_file
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.losses import categorical_crossentropy

#from scipy import misc
#from scipy.misc import imread, imresize, imsave
#from imageio import imread
#from scipy.misc import imresize, imsave
from cv2 import imread, resize, imwrite

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



class CATH_Classifier():
    ALLOWED_MODELS = ["none", "unet","unet2", "fullconv","lenet", "lenet2", "lenet3", "vgg16", "vgg19", 
                      "inception", "xception", "resnet50", "mobilenet", "densenet121", 
                      "densenet169", "densenet201", "inception_resnetv2", "nasnetlarge", "nasnetmobile"]
    
    def __init__(self, root_path, image_folder="png", img_dim=-1, batch_size=16, epochs=10, 
                 model="lenet", data_labels_filename="cath-dataset-nonredundant-S40.txt", label_columns="2,3,4,5", 
                 png_suffix="_rgb.png", nchannels=3, sample_size=None, selection_filename=None, idlength=7, 
                 kernel_shape="3,3", dim_ordering="channels_last", valsplit=0.33, save_resized_images=False, 
                 outdir="results", use_pretrained=False, early_stopping="none", tensorboard=False, tbdir='./tblog', optimizer="adam", 
                 verbose=True, batch_norm=True, act="relu", learning_rate=-1, img_size_bins=[], dropout=0.25, 
                 img_bin_batch_sizes=[], flipdia_img=False, inverse_img=False, model_arch_file=None, 
                 model_weights_file=None, fc1=512, fc2=512, keras_top=True, ngpus=1, train_verbose=True, 
                 generic_label="kcc", show_images=False, img_bin_clusters=3, min_img_size=-1, max_img_size=-1,
                 h5_backend="h5py", h5_input="", use_dc_decoder=False, dc_dec_weight=1.0, dc_dec_act="relu", dc_decoder_loss="mean_squared_logarithmic_error",
                 use_seq_encoder=False, use_seq_decoder=False,seq_decoder_loss="mean_squared_logarithmic_error", seq_dec_weight=1.0, seq_code_layer="fc_first", seq_dec_act="relu", manual_seq_maxlength=None, seq_enc_arch="cnn",
                 cath_loss_weight=1.0, checkpoints=10, CATH_Y_labels_interpretable=False, cath_domlength_weight=1.0, domlength_regression_loss="mean_squared_logarithmic_error"):
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
        assert os.path.exists(self.data_labels_filename), self.data_labels_filename
        
        if not label_columns=="":
            self.label_columns = list(map(int, label_columns.split(",")) )
            
        else:
            self.label_columns = []
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
        self.kernel_shape = list(map(int, tuple(kernel_shape.split(","))))
        self.img_bin_clusters = img_bin_clusters
        K.set_image_data_format(dim_ordering)
        print(K.image_data_format())
        print(K.image_dim_ordering())
        self.fc1 = fc1
        self.fc2 = fc2
        self.keras_top = keras_top
        self.valsplit = valsplit
        self.save_resized_images = save_resized_images
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        self.use_pretrained = use_pretrained
        self.early_stopping = early_stopping
        self.tensorboard = tensorboard
        self.tbdir = tbdir
        self.optimizer = optimizer
        self.history = None
        self.verbose = verbose
        self.batch_norm = batch_norm
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
        self.X = None
        self.Ys = None
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
        self.seqdata = None
        self.seq_maxlength = -1
        self.manual_seq_maxlength = manual_seq_maxlength
        self.seq_enc_arch = seq_enc_arch
        assert seq_enc_arch in ["cnn","lstm"]
        
        self.min_img_size = min_img_size
        self.max_img_size = max_img_size
        
        self.use_dc_decoder = use_dc_decoder
        self.dc_dec_weight = dc_dec_weight
        self.dc_dec_act = dc_dec_act
        self.dc_decoder_loss = dc_decoder_loss
        
        self.use_seq_encoder = use_seq_encoder
        self.use_seq_decoder = use_seq_decoder and use_seq_encoder
        self.seq_dec_weight = seq_dec_weight
        self.seq_code_layer = seq_code_layer
        self.seq_dec_act = seq_dec_act
        self.seq_decoder_loss = seq_decoder_loss
        
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
        
    def getTrainTestSetMulti(self, resized_img_loc=""):
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
                line = l.decode("utf-8-sig")
                if line[0] != '#':
                    tmp = line.strip().split()
                    assert len(tmp) == 12, str(tmp)
                    if self.selection_filename != None and not tmp[0].strip() in mydoms:
                        continue
                    else:
                        dom_dict[tmp[0]] = tmp[1:]
        img_data = []
        classes = []
        labels = []
        orig_sizes = []
        img_data_s = {s:[] for s in self.img_size_bins}
        classes_s = {s:[] for s in self.img_size_bins}
        labels_s = {s:[] for s in self.img_size_bins}
        
        found_count = 0
        not_found_list = []
        files = glob.glob(self.image_folder + "/" + "???????%s" % self.png_suffix)
        
        class_dict = self.get_class_dict()# get the intepretable names of domains
        
        ioerror_list = []
        
        if self.sample_size != None:
            files = np.random.choice(files, self.sample_size)
        print ("Reading %i image files..." % (len(files)))
        MLhelpers.print_progress(0, len(files))
        for fi in range(len(files)):
            f = files[fi]
            cath_code = os.path.basename(f)[:7]
            if cath_code in dom_dict:
                if self.CATH_Y_labels_interpretable:
                    pred_classes = [ class_dict[".".join(
                            [ dom_dict[cath_code][j - 2] for j in range(self.label_columns[0], i + 1) ]
                        )]["name"]  
                            for i in self.label_columns]
                    if self.verbose: print (cath_code, pred_classes)
                else:
                    #pred_classes = [dom_dict[cath_code][i-2] for i in self.label_columns]
                    pred_classes = []
                    for L in self.label_columns:
                        if L in [2,3,4,5]:
                            pred_classes.append( ".".join(
                                    [ dom_dict[cath_code][j - 2] for j in range(self.label_columns[0], L + 1) ]
                                ) )
                        else:
                            pred_classes.append(dom_dict[cath_code][L - 2])
                                
                    if self.verbose: print (cath_code, pred_classes, [class_dict[p] for p in pred_classes[:3]])
                try:
                    
                    if self.nchannels == 3:
                        
                        img = imread(f, flatten=False, mode="RGB")
                        orig_size = img.shape[1]
                        if self.min_img_size != -1 and orig_size < self.min_img_size: continue
                        if self.max_img_size != -1 and orig_size > self.max_img_size: continue
                        # print (img.shape)
                        # img = image.load_img(f, target_size=(outdim, outdim), grayscale=False)
                        if self.img_dim != None and self.img_dim != img.shape[1]:
                            img = resize(img, size=(self.img_dim, self.img_dim), mode="RGB")
                        x = np.array(img)
                        if K.image_data_format() == "channels_first":
                            x = np.reshape(img, (3, self.img_dim, self.img_dim))
                    else:   
                        img = imread(f, flatten=True)
                        orig_size = img.shape[1]
                        if self.min_img_size != -1 and orig_size < self.min_img_size: continue
                        if self.max_img_size != -1 and orig_size > self.max_img_size: continue
                        # print (img.shape)
                        # img = image.load_img(f, target_size=(outdim, outdim), grayscale=True)
                       
                        if self.img_dim != None and self.img_dim != img.shape[1]:
                            img = resize(img, size=(self.img_dim, self.img_dim), mode="L")
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
                        imwrite(os.path.join(resized_img_loc, os.path.basename(f)), tmp, "png")
                    
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
                        labels.append(cath_code)
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
                not_found_list.append(cath_code)
            MLhelpers.print_progress(fi, len(files))
        print ("Found %i image files." % len(img_data))
        
        with open("ioerrors.txt", "w") as iof:
            iof.write("\n".join(ioerror_list) + "\n")
        # print (np.array(img_data).reshape((len(img_data),32,32,3)))
        # print (len(img_data))
        
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
            
            self.X = np.array(img_data)
            self.Ys = list(np.array(classes).T)
            self.labels = labels
            self.not_found = not_found_list
        else:
            self.dX = {s:np.array(img_data_s[s]) for s in self.img_size_bins}
            self.dYs = {s:np.array(classes_s[s]).T for s in self.img_size_bins}
            self.dlabels = {s:labels_s[s] for s in self.img_size_bins}
            self.not_found = not_found_list

    def get_sequence_data(self):

        with open(os.path.join(self.root_path, "cath-domain-seqs.fa")) as fastafile:
            for entry in fastafile.read().split(">")[1:]:
                try:
                    tmp = entry.split("\n")
                    meta = tmp[0]
                    # startstop = meta.split("/")[1].split("_") 
                    
                    cathid = meta.split("|")[2].split("/")[0]
                    sequence = tmp[1]
                    self.seq_maxlength = max(self.seq_maxlength, len(sequence))
                    # print( cathid, startstop, sequence)
                    self.seqdict[cathid] = sequence
                except ValueError as ve:
                    print (ve)
                    print (entry)
    
                    
    def softMaxAxis(self, axis):
        def soft(x):
            return softmax(x, axis=axis)
        return soft


    def prepare_sequence_data(self):
        
        if self.manual_seq_maxlength != None:
            input_seqlength = self.manual_seq_maxlength
        else:
            input_seqlength = self.seq_maxlength
        
        if type(self.labels) != type(None):
            if self.seqdict == {} or self.seq_maxlength == -1:
                self.get_sequence_data()
                assert self.seqdict != {}
                assert self.seq_maxlength != -1
                
            chars = 'ACDEFGHIKLMNPQRSTVWYX'
            aa2num = {}
            for a in range(len(chars)):
                zeros = np.zeros(len(chars) + 1)
                # print ()
                zeros[1:] = np.array([1 if c == chars[a] else 0 for c in chars])
                aa2num[chars[a]] = zeros
                # print (a, chars[a], zeros)
            # aa2num = {chars[a]:np.zeros(len(chars)+1) for a in range(len(chars))}
            
            seq_enc = []
        
            for cath_code in self.labels:
                if cath_code in self.seqdict:   
                    seq = self.seqdict[cath_code]
                    tmparr = np.zeros((min(input_seqlength, len(seq)), len(chars) + 1))
                    tmparr[:, 0] = 1.0
                    tmparr[0:min(len(seq), input_seqlength), ] = np.array([aa2num[seq[l]] for l in range(min(len(seq), self.seq_maxlength))])
                    
                    seq_enc.append(tmparr)
                else:
                    tmparr = np.zeros((input_seqlength, len(chars) + 1))
                    tmparr[:, 0] = 1.0
                    seq_enc.append(tmparr)
                    # print(cath_code,seq,seq_enc[-1], seq_enc[-1].shape)
            self.seqdata = sequence.pad_sequences(seq_enc, maxlen=input_seqlength)
    
    def round_figures(self, x, n):
        if x == 0: return None
        """Returns x rounded to n significant figures."""
        return round(x, int(n - math.ceil(math.log10(abs(x)))))

    def plot_curves(self, metric="loss"):
        if self.history == None: return
        cath_label_map = {"Cout":"C", "Aout":"A", "Tout":"T", "Hout":"H"}
        
        # rf_metric = "acc"#"f1"#"f1"
        train_metric = metric  # "fbeta_score"#"fbeta_score"
        # test_metric = "val_"+train_metric
        color_map = {"Cout":"blue", "Aout":"green", "Tout":"red", "Hout":"cyan"}
        # fig = plt.subplots(2,2)
        
        all_train = {}
        all_test = {}
        d = self.history.history
        print (d.keys())
        rf_val = None
            
        actual_ep = len(d["loss"])
        print ("found %i epochs" % (actual_ep))
        
        for out in ["Cout", "Aout", "Tout", "Hout"]:
            if out == "Cout" and not 2 in self.label_columns: continue
            if out == "Aout" and not 3 in self.label_columns: continue
            if out == "Tout" and not 4 in self.label_columns: continue
            if out == "Hout" and not 5 in self.label_columns: continue
            
            train_label = out + "_" + train_metric
            test_label = "val_" + out + "_" + train_metric
            
            col = color_map[out]

            if train_metric == "loss":
                print("min(%s) for training (validation) '%s' = %.3f (%.3f)" % (train_label, cath_label_map[out], min(d[train_label]), min(d[test_label])))
            else:
                print("max(%s) for training (validation) '%s' = %.3f (%.3f)" % (train_label, cath_label_map[out], max(d[train_label]), max(d[test_label])))
            
            plt.plot(d[train_label], linestyle="-", color=col, label=cath_label_map[out])
            plt.plot(d[test_label], linestyle=":", color=col, label=None)
            
            if d[train_label][-1] != 0: plt.annotate(str(self.round_figures(d[train_label][-1], 3)) + "(train)", (actual_ep, d[train_label][-1]), xycoords='data', color=col)
            if d[test_label][-1] != 0: plt.annotate(str(self.round_figures(d[test_label][-1], 3)) + "(test)", (actual_ep, d[test_label][-1]), xycoords='data', color=col)
            
            all_train[out] = d[out + "_" + train_metric]
            all_test[out] = d["val_" + out + "_" + train_metric]
        
        if rf_val != None: 
            plt.plot([rf_val for i in range(self.epochs)], linestyle="--", color="black", label=None)
            plt.annotate(str(self.round_figures(rf_val, 3)) + "(RF)", (actual_ep, rf_val), xycoords='data', color="black")
        plt.xlabel("epoch")
        plt.ylabel("metric:" + train_metric)
        plt.title("CNN - 4 output; img dim: %sx%s (filter %ix%i)\n'-'=train(67%%); '...'=test(33%%); '--'=RF control" % (str(self.img_dim), str(self.img_dim), self.kernel_shape[0], self.kernel_shape[1]))
        max_sum = None  
        max_sum_t = None             
        for t in range(actual_ep):
            s_train = 0
            s_test = 0
            for out in ["Cout", "Aout", "Tout", "Hout"]:
                if out == "Cout" and not 2 in self.label_columns: continue
                if out == "Aout" and not 3 in self.label_columns: continue
                if out == "Tout" and not 4 in self.label_columns: continue
                if out == "Hout" and not 5 in self.label_columns: continue
                s_train += all_train[out][t]
                s_test += all_test[out][t]
            s = s_train + s_test
            if s > max_sum:
                max_sum = s
                max_sum_t = t
        
        print("Best t = %i (score=%.3f)" % (max_sum_t, max_sum / 8))
            
        if train_metric == "loss":
            plt.legend(loc="upper right")
        else:
            plt.legend(loc="lower right")
        
        # pngname = str(fullpath).replace(".json","_%s_plot.png"%(train_metric))
        # print( pngname )
        # plt.savefig(pngname)
        plt.show()
    
    def lenet(self, X, nb_classes, batch_norm=True, kernel_init=he_normal(), 
              bias_init=he_normal(), act="relu", f2=(3, 3), padding='same',
              conv_strides=(1,1),pool_strides=(2,2)):
        
        
        if self.img_dim%2==0:
            f2=(3,3)
        else:
            f2=(2,2)
        print ("kernel shapes:",self.kernel_shape, f2)

        print ("lenet input:", X.shape)
        if K.image_data_format == "channels_first":
            inshape = (self.nchannels, self.img_dim, self.img_dim)
        else:
            inshape = (self.img_dim, self.img_dim, self.nchannels)
        main_input = Input(shape=(inshape), dtype='float32', name='main_input')

        # bshape = tuple(None)+X.shape[1:]
        
        # main_input = Input(batch_shape=bshape, dtype='float32', name='main_input')
        channel_format = K.image_data_format()
        
        x = Conv2D(32, self.kernel_shape,
                   strides=conv_strides,
                                padding=padding,
                                data_format=channel_format,
                                kernel_initializer=kernel_init, bias_initializer=bias_init,
                                name='lenet_conv2d_1')(main_input)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x) 
        x = Conv2D(32, self.kernel_shape, 
                   strides=conv_strides,
                   padding=padding,
                   kernel_initializer=kernel_init, bias_initializer=bias_init,
                    name='lenet_conv2d_2')(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        
        x = AveragePooling2D(pool_size=(2, 2), strides=pool_strides, padding='same', data_format=channel_format)(x)
        #x = Dropout(self.dropout)(x)
    
        x = Conv2D(64, f2, 
                   strides=conv_strides,
                   padding=padding,
                    kernel_initializer=kernel_init, bias_initializer=bias_init,
                    name='lenet_conv2d_3')(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = Conv2D(64, f2, 
                   strides=conv_strides,
                   padding=padding,
                   kernel_initializer=kernel_init, bias_initializer=bias_init,
                    name='lenet_conv2d_4')(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        
        x = last_maxpool = AveragePooling2D(pool_size=(2, 2), strides=pool_strides, padding='same', data_format=channel_format, 
                                        name='last_maxpool')(x)
        #x = Dropout(self.dropout)(last_maxpool)
    
        if self.img_dim == None:
            x = GlobalAveragePooling2D()(x)
        else:
            x = Flatten()(x)

        x, added_inputs = self.add_fclayers_to_model(x)
        
        inputs = [main_input] + added_inputs
        print ("model inputs:", inputs)
        
        outputs = self.add_outputs_to_model(x, maxpool2D=last_maxpool)
        
        model = Model(inputs=inputs, outputs=outputs)        
        return model

    def lenet2(self, X, nb_classes, batch_norm=True, kernel_init=he_normal(), bias_init=he_normal(), act="relu"):
        print ("lenet2 input:", X.shape)
        if K.image_data_format == "channels_first":
            inshape = (self.nchannels, self.img_dim, self.img_dim)
        else:
            inshape = (self.img_dim, self.img_dim, self.nchannels)
        main_input = Input(shape=(inshape), dtype='float32', name='main_input')
        channel_format = K.image_data_format()
        
        x = Conv2D(32, self.kernel_shape,
                    padding='same',
                    data_format=channel_format,
                    kernel_initializer=kernel_init, bias_initializer=bias_init)(main_input)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        x = Conv2D(32, self.kernel_shape,
                   padding='same',
                    data_format=channel_format,
                    kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=channel_format)(x)
        x = Dropout(self.dropout)(x)
    
        x = Conv2D(64, self.kernel_shape, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        x = Conv2D(64, self.kernel_shape, kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        x = Conv2D(64, self.kernel_shape, kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=channel_format)(x)
        x = Dropout(self.dropout)(x)
        
        x = Conv2D(128, self.kernel_shape, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        x = Conv2D(128, self.kernel_shape, kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        x = Conv2D(128, self.kernel_shape, kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=channel_format)(x)
        x = Dropout(self.dropout)(x)
    
        x = Flatten()(x)

        x, added_inputs = self.add_fclayers_to_model(x)
        
        inputs = [main_input] + added_inputs
        print ("model inputs:", inputs)
        
        outputs = self.add_outputs_to_model(x)
        
        model = Model(inputs=inputs, outputs=outputs)        
        return model    

    def lenet3(self, X, nb_classes, batch_norm=True, kernel_init=he_normal(), bias_init=he_normal(), act="relu", dropout=0.25, fc=512, conv1=128, conv2=256, f2=(3, 3)):
        if self.dropout > 0: dropout = self.dropout
        f1 = self.kernel_shape
        print ("lenet3 input:", X.shape)
        if K.image_data_format == "channels_first":
            inshape = (self.nchannels, self.img_dim, self.img_dim)
        else:
            inshape = (self.img_dim, self.img_dim, self.nchannels)
        main_input = Input(shape=(inshape), dtype='float32', name='main_input')        
        channel_format = K.image_data_format()
        
        x = Conv2D(conv1, f1, padding='same', data_format=channel_format,
                                kernel_initializer=kernel_init, bias_initializer=bias_init)(main_input)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x) 
        x = Conv2D(conv1, f1, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=channel_format)(x)
        x = Dropout(dropout)(x)
    
        x = Conv2D(conv2, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = Conv2D(conv2, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=channel_format)(x)
        x = Dropout(dropout)(x)
        
        if self.img_dim == None:
            x = GlobalAveragePooling2D()(x)
        else:
            x = Flatten()(x)
        
        x, added_inputs = self.add_fclayers_to_model(x)
        
        inputs = [main_input] + added_inputs
        print ("model inputs:", inputs)
        
        outputs = self.add_outputs_to_model(x)
        
        model = Model(inputs=inputs, outputs=outputs)        
        
        return model    

    def dc_decoder(self, encoded, act="relu", batch_norm=False, kernel_init=he_normal(), bias_init=he_normal(), 
        channel_format=K.image_data_format(), f2=(3, 3), use_dropout=False, out_act='relu'):
        
        print("encoded", encoded.shape)
        # x = Dense(3136)(encoded)
        # x = Reshape((4,4,32))(encoded)
        encoded_length = encoded.get_shape().as_list()[1]
        
        reshape_dim = 1
        reshape_chan = encoded_length / reshape_dim ** 2
        
        print(reshape_dim, reshape_chan)
        x = Reshape((reshape_dim, reshape_dim, reshape_chan))(encoded)
        print("reshape", x.shape)
        x = UpSampling2D(size=(2, 2), data_format=channel_format)(x)
        
        x = Conv2D(200, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(200, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        print("conv2d", x.shape)
        x = UpSampling2D(size=(7, 7), data_format=channel_format)(x)
        print("up2d", x.shape)
        
        x = Conv2D(100, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(100, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        print("conv2d", x.shape)
        x = UpSampling2D(size=(4, 4), data_format=channel_format)(x)
        print("up2d", x.shape)

        x = Conv2D(50, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(50, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        print("conv2d", x.shape)
        x = UpSampling2D(size=(4, 4), data_format=channel_format)(x)
        print("up2d", x.shape)        

        decoded = Conv2D(self.nchannels, (3, 3), activation=out_act, padding='same', name="dc_decoder")(x)
        print("dc_decoded", decoded.shape)
        return decoded
    
    
    def dc_decoder2(self, encoded, act="relu", batch_norm=False, kernel_init=he_normal(), bias_init=he_normal(), 
        channel_format=K.image_data_format(), f2=(3, 3), use_dropout=False, out_act='relu'):
        
        print("encoded", encoded.shape)
        # x = Dense(3136)(encoded)
        # x = Reshape((4,4,32))(encoded)
        encoded_length = encoded.get_shape().as_list()[1]
        
        reshape_dim = 1
        reshape_chan = encoded_length / reshape_dim ** 2
        
        print(reshape_dim, reshape_chan)
        x = Reshape((reshape_dim, reshape_dim, reshape_chan))(encoded)
        print("reshape", x.shape)
        
        
        nn=1024
        print("conv2d", x.shape)
        x = UpSampling2D(size=(7, 7), data_format=channel_format)(x)
        print("up2d", x.shape)
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)

        
        nn=786
        print("conv2d", x.shape)
        x = UpSampling2D(size=(2, 2), data_format=channel_format)(x)
        print("up2d", x.shape)
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        nn=512
        print("conv2d", x.shape)
        x = UpSampling2D(size=(2, 2), data_format=channel_format)(x)
        print("up2d", x.shape)
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        
        nn=256
        print("conv2d", x.shape)
        x = UpSampling2D(size=(2, 2), data_format=channel_format)(x)
        print("up2d", x.shape)
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)

        nn=128
        print("conv2d", x.shape)
        x = UpSampling2D(size=(2, 2), data_format=channel_format)(x)
        print("up2d", x.shape)
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        nn=64
        print("conv2d", x.shape)
        x = UpSampling2D(size=(2, 2), data_format=channel_format)(x)
        print("up2d", x.shape)
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(nn, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
       
        decoded = Conv2D(self.nchannels, 1, activation=out_act, padding='same', name="dc_decoder")(x)
        print("dc_decoded", decoded.shape)
        return decoded
    def dc_decoder_from_2D_lenet(self, encoded, act="relu", batch_norm=False, kernel_init=he_normal(), bias_init=he_normal(), channel_format=K.image_data_format(), f2=(3, 3), use_dropout=False, out_act='sigmoid'):
        
        if self.img_dim%2==0:
            f2=(3,3)
        else:
            f2=(2,2)
        
        print ("kernel shapes:",self.kernel_shape, f2)
        print("encoded", encoded.shape)
        print ("encoded shape:",encoded.shape)
        
        x = Conv2D(64, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(encoded)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(64, f2, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        print("conv2d", x.shape)
        x = UpSampling2D(size=(2, 2), data_format=channel_format)(x)
        print("up2d", x.shape)
        
        x = Conv2D(32, self.kernel_shape, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        x = Conv2D(32, self.kernel_shape, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation(act)(x)
        if use_dropout: x = Dropout(self.dropout)(x)
        
        print("conv2d", x.shape)
        x = UpSampling2D(size=(2, 2), data_format=channel_format)(x)
        print("up2d", x.shape)

        
        decoded = Conv2D(self.nchannels, self.kernel_shape, activation=out_act, padding='same', name="dc_decoder")(x)
        print("dc_decoded", decoded.shape)
        return decoded

    def lstm_seq_encoder(self, act='relu', hidden_length=128):
        print ("LSTM ENCODER")
        lstm_input = Input(shape=(self.manual_seq_maxlength,22,), dtype='float32', name='lstm_input')
        print ("lstm input:",lstm_input.shape)
        #lstm_emb = Embedding(output_dim=64, input_dim=22, input_length=self.manual_seq_maxlength)(lstm_input)
        # ("lstm emb:",lstm_emb.shape)
        lstm_enc = LSTM(hidden_length, activation=act, return_sequences=True)(lstm_input)
        print ("lstm enc:",lstm_enc.shape)

        return lstm_enc, lstm_input
    
    def lstm_seq_decoder(self, encoded, act='relu'):
        print ("lstm_seq_decoder: encoded",encoded.shape)
        lstm_dec = LSTM(22, activation=act,return_sequences=True, name="seq_dec")(encoded)
        print ("lstm_seq_decoder: decoded",encoded.shape)
        return lstm_dec
    
    def seqnet(self):
        if self.manual_seq_maxlength != None:
            input_seqlength = self.manual_seq_maxlength
        else:
            input_seqlength = self.seq_maxlength
        
        print ("input_seqlength=", input_seqlength)
        
        seq_input = Input(shape=(input_seqlength, 22), dtype='float32', name='seq_input')
        print ("seqinput", seq_input.shape)

        #input_size = (self.img_dim,self.img_dim,self.nchannels)
        #inputs = Input(input_size, dtype='float32', name='main_input')
        conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_input)
        conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling1D(pool_size=2)(conv3)
        conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling1D(pool_size=2)(drop4)
    
        conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
    
        up6 = Conv1D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(drop5))
        merge6 = concatenate([drop4,up6], axis = 2)
        conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
        up7 = Conv1D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv6))
        merge7 = concatenate([conv3,up7], axis = 2)
        conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
        up8 = Conv1D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv7))
        merge8 = concatenate([conv2,up8], axis = 2)
        conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
        up9 = Conv1D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv8))
        merge9 = concatenate([conv1,up9], axis = 2)
        conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv1D(22, 1, activation = 'sigmoid', name="seq_decoder")(conv9)
        
        

        flat = GlobalAveragePooling1D()(conv5)

        x, added_inputs = self.add_fclayers_to_model(flat)
        
        inputs = [seq_input] + added_inputs
        print ("model inputs:", inputs)
        
        outputs = self.add_outputs_to_model(x, unet_decoder=conv10)
        
        model = Model(inputs=inputs, outputs=outputs)        
        #model = Model(input = inputs, output = conv10)
    
        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #model.summary()
    
        
        return model    
    def dc_seq_encoder(self, act="relu", batch_norm=False, kernel_init=he_normal(), bias_init=he_normal(), channel_format=K.image_data_format(), f2=3, use_dropout=False):
        
        if self.manual_seq_maxlength != None:
            input_seqlength = self.manual_seq_maxlength
        else:
            input_seqlength = self.seq_maxlength
        
        print ("input_seqlength=", input_seqlength)
        
        seq_input = Input(shape=(input_seqlength, 22), dtype='float32', name='seq_input')
        print ("seqinput", seq_input.shape)
        
        seq = Conv1D(32, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(seq_input)
        if batch_norm: seq = BatchNormalization()(seq)
        seq = Activation(act)(seq) 
        if use_dropout: seq = Dropout(self.dropout)(seq)

        seq = Conv1D(32, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(seq)
        if batch_norm: seq = BatchNormalization()(seq)
        seq = Activation(act)(seq) 
        if use_dropout: seq = Dropout(self.dropout)(seq)

        print("seq", seq.shape)
        seq = MaxPooling1D(2, padding='same')(seq)
        print("seqpool", seq.shape)
        
        seq = Conv1D(48, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(seq)
        if batch_norm: seq = BatchNormalization()(seq)
        seq = Activation(act)(seq) 
        if use_dropout: seq = Dropout(self.dropout)(seq)

        seq = Conv1D(48, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(seq)
        if batch_norm: seq = BatchNormalization()(seq)
        seq = Activation(act)(seq) 
        if use_dropout: seq = Dropout(self.dropout)(seq)

        print("seq", seq.shape)
        seq = MaxPooling1D(2, padding='same')(seq)
        print("seqpool", seq.shape)

        seq = Conv1D(64, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(seq)
        if batch_norm: seq = BatchNormalization()(seq)
        seq = Activation(act)(seq) 
        if use_dropout: seq = Dropout(self.dropout)(seq)

        seq = Conv1D(64, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(seq)
        if batch_norm: seq = BatchNormalization()(seq)
        seq = Activation(act)(seq) 
        if use_dropout: seq = Dropout(self.dropout)(seq)

        print("seq", seq.shape)
        seq = MaxPooling1D(2, padding='same')(seq)
        print("seqpool", seq.shape)
        
        seq = Flatten(name="seq_flat")(seq)
        print("seqflat", seq.shape)
        return seq, seq_input

    def dc_seq_decoder(self, encoded, act="relu", batch_norm=False, kernel_init=he_normal(), bias_init=he_normal(), channel_format=K.image_data_format(), f2=(3, 3), use_dropout=False, out_act='sigmoid'):
        print("encoded", encoded.shape)

        encoded_length = encoded.get_shape().as_list()[1]
        print ("encoded length:", encoded_length)
        if encoded_length == 400:
            reshape_factor = 2
        elif encoded_length == 800:
            reshape_factor = 4
        elif encoded_length == 1600:
            reshape_factor = 8
        else:
            reshape_factor = 1600 / 200
        
        print ("reshape factor:", reshape_factor)
        dseq = Reshape((encoded_length / reshape_factor, reshape_factor))(encoded)
        print ("dseq reshape", dseq.shape)
        
        dseq_act = act
        
        dseq = Conv1D(64, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(dseq)
        if batch_norm: dseq = BatchNormalization()(dseq)
        dseq = Activation(dseq_act)(dseq)
        if use_dropout: dseq = Dropout(self.dropout)(dseq)        

        dseq = Conv1D(64, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(dseq)
        if batch_norm: dseq = BatchNormalization()(dseq)
        dseq = Activation(dseq_act)(dseq)
        if use_dropout: dseq = Dropout(self.dropout)(dseq)     
        
        print("dseq", dseq.shape)
        dseq = UpSampling1D(2)(dseq)
        print("dsequp", dseq.shape)
        
        dseq = Conv1D(32, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(dseq)
        if batch_norm: dseq = BatchNormalization()(dseq)
        dseq = Activation(dseq_act)(dseq)
        if use_dropout: dseq = Dropout(self.dropout)(dseq)        

        dseq = Conv1D(32, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(dseq)
        if batch_norm: dseq = BatchNormalization()(dseq)
        dseq = Activation(dseq_act)(dseq)
        if use_dropout: dseq = Dropout(self.dropout)(dseq)     
        
        print("dseq", dseq.shape)
        dseq = UpSampling1D(2)(dseq)
        print("dsequp", dseq.shape)
        
        dseq = Conv1D(22, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(dseq)
        if batch_norm: dseq = BatchNormalization()(dseq)
        dseq = Activation(dseq_act)(dseq)
        if use_dropout: dseq = Dropout(self.dropout)(dseq)        

        dseq = Conv1D(22, 3, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(dseq)
        if batch_norm: dseq = BatchNormalization()(dseq)
        dseq = Activation(dseq_act)(dseq)
        if use_dropout: dseq = Dropout(self.dropout)(dseq)     
        
        print("dseq", dseq.shape)
        dseq = UpSampling1D(2)(dseq)
        print("dsequp", dseq.shape)
        
        dseq = Conv1D(22, 3, activation=out_act, name="seq_dec", padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(dseq)
        
        print("seq_decoded", dseq.shape)
        return dseq


    def unet(self):
        input_size = (self.img_dim,self.img_dim,self.nchannels)
        inputs = Input(input_size, dtype='float32', name='main_input')
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
    
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation = 'linear', name="dc_decoder")(conv9)
        
        
        flat = Flatten()(pool4)

        x, added_inputs = self.add_fclayers_to_model(flat)
        
        inputs = [inputs] + added_inputs
        print ("model inputs:", inputs)
        
        outputs = self.add_outputs_to_model(x, unet_decoder=conv10)
        
        model = Model(inputs=inputs, outputs=outputs)        
        #model = Model(input = inputs, output = conv10)
    
        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #model.summary()
    
        
        return model

    def unet2(self):
        input_size = (self.img_dim,self.img_dim,self.nchannels)
        inputs = Input(input_size, dtype='float32', name='main_input')
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
    
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        #merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        #merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        #merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation = 'linear', name="dc_decoder")(conv9)
        
        
        flat = Flatten()(pool4)

        x, added_inputs = self.add_fclayers_to_model(flat)
        
        inputs = [inputs] + added_inputs
        print ("model inputs:", inputs)
        
        outputs = self.add_outputs_to_model(x, unet_decoder=conv10)
        
        model = Model(inputs=inputs, outputs=outputs)        
        #model = Model(input = inputs, output = conv10)
    
        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #model.summary()
    
        
        return model


    def add_fclayers_to_model(self, x):
        added_inputs = []
        # let's add a fully-connected layer
        if self.fc1 > 0:
            if self.use_seq_encoder and self.seq_code_layer == "fc_first":
                if self.seq_enc_arch=="cnn":
                    seq_enc, seq_input = self.dc_seq_encoder(act="relu", batch_norm=True, kernel_init=he_normal(), bias_init=he_normal(), channel_format=K.image_data_format(), f2=3, use_dropout=False)
                else:
                    seq_enc, seq_input = self.lstm_seq_encoder(act="relu")

                added_inputs.append(seq_input)
                
                seq_fc = Dense(self.fc1, kernel_regularizer=l2(0.01))(seq_enc)
                if self.batch_norm: seq_fc = BatchNormalization()(seq_fc)
                seq_fc = Activation(self.act, name="fc_seq")(seq_fc)
                #seq_fc = Dropout(self.dropout)(seq_fc)
                
                x = Dense(self.fc1, kernel_regularizer=l2(0.01))(x)
                if self.batch_norm: x = BatchNormalization()(x)
                x = Activation(self.act, name="fc_first")(x)
                # x = Dropout(self.dropout)(x)
                
                x = average([x, seq_fc])
                # x = concatenate([x, seq_fc], 1)
                #if self.batch_norm: x = BatchNormalization()(x)
                #x = Activation(self.act, name="fc_merge")(x)
                x = Dropout(self.dropout)(x)
            else:
                x = Dense(self.fc1)(x)
                if self.batch_norm: x = BatchNormalization()(x)
                x = Activation(self.act, name="fc_first")(x)
                x = Dropout(self.dropout)(x)
                
        if self.fc2 > 0:
            if self.use_seq_encoder and self.seq_code_layer == "fc_last":
                if self.seq_enc_arch=="cnn":
                    seq_enc, seq_input = self.dc_seq_encoder(act="relu", batch_norm=True, kernel_init=he_normal(), bias_init=he_normal(), channel_format=K.image_data_format(), f2=3, use_dropout=False)
                else:
                    seq_enc, seq_input = self.lstm_seq_encoder(act="relu")
                added_inputs.append(seq_input)
                
                seq_fc = Dense(self.fc2, kernel_regularizer=l2(0.01))(seq_enc)
                if self.batch_norm: seq_fc = BatchNormalization()(seq_fc)
                seq_fc = Activation(self.act, name="fc_seq")(seq_fc)
                # seq_fc = Dropout(self.dropout)(seq_fc)
                
                x = Dense(self.fc2, kernel_regularizer=l2(0.01))(x)
                if self.batch_norm: x = BatchNormalization()(x)
                x = Activation(self.act, name="fc_last-1")(x)
                # x = Dropout(self.dropout)(x)
                
                x = average([x, seq_fc])  # now the merged layer becomes the last layer!
                # x = concatenate([x, seq_fc], 1)
                #if self.batch_norm: x = BatchNormalization()(x)
                #x = Activation(self.act, name="fc_last")(x)
                x = Dropout(self.dropout)(x)
            else:
                x = Dense(self.fc2)(x)
                if self.batch_norm: x = BatchNormalization()(x)
                x = Activation(self.act, name="fc_last")(x)
                x = Dropout(self.dropout)(x)

        return x, added_inputs
    
    def add_outputs_to_model(self, x, maxpool2D=None, unet_decoder=None, seqnet_decoder=None):
        # x is the incoming layer from an existing model
        
        # collect outputs
        outputs = []

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
            LENout = Dense(1, activation="linear", name="LENout")(x)
            outputs.append(LENout)
        if 12 in self.label_columns:
            RESout = Dense(1, activation="linear", name="RESout")(x)
            outputs.append(RESout)
        
        # optionally, add a 2D decoder
        if self.use_dc_decoder:
            if self.model == "lenet":
                print ("Connecting DC decoder to last maxpool layer...")
                dcdec = self.dc_decoder_from_2D_lenet(maxpool2D, act=self.dc_dec_act, batch_norm=True)
            if self.model in ["unet", "fullconv"]:
                dcdec = unet_decoder
            else:
                dcdec = self.dc_decoder2(x, act=self.dc_dec_act, batch_norm=False)
            outputs.append(dcdec)
        
        if self.use_seq_decoder:
            if self.model=="fullconv":
                seqdec = seqnet_decoder
                outputs.append( seqdec  )
                
            else:
                if self.seq_enc_arch=='cnn':
                    seqdec = self.dc_seq_decoder(x, act=self.seq_dec_act, batch_norm=True)
                else:
                    seqdec = self.lstm_seq_decoder(x, act=self.seq_dec_act)
                outputs.append(seqdec)

        return outputs


    def get_fullconv_model(self, branch_2d=True, branch_1d=True):
        assert branch_2d or branch_1d
        inputs = []
        if branch_2d:
            input_size = (self.img_dim, self.img_dim, self.nchannels)
            input = Input(input_size, dtype='float32', name='main_input') # 224
            inputs.append(input)
            conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
            conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 112
            conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
            conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 56
            conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
            conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 28
            conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
            conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
            drop4 = Dropout(self.dropout)(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # 14
    
            conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
            conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
            drop5 = Dropout(self.dropout)(conv5)
    
            up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
            merge6 = concatenate([drop4,up6], axis = 3)
            conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
            conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
            up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
            merge7 = concatenate([conv3,up7], axis = 3)
            conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
            conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
            up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
            merge8 = concatenate([conv2,up8], axis = 3)
            conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
            conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
            up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
            merge9 = concatenate([conv1,up9], axis = 3)
            conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
            conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
            conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
            conv10 = Conv2D(3, 1, activation = 'linear', name="dc_decoder")(conv9)
        


            # branch off 14x14 layer, leads to classification task
            pool_down1 = AveragePooling2D(pool_size=(2, 2))(drop5) # 7
            conv_down1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool_down1)
            conv_down2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_down1)
            drop_down1 = Dropout(self.dropout)(conv_down2)

            final_pool = GlobalAveragePooling2D()(drop_down1)
            print("final_pool 2D", final_pool.shape)
            #final = GlobalAveragePooling1D(name="fc_last")(final_pool) ### TODO: FIX ME - wrong dims
            #x, added_inputs = self.add_fclayers_to_model(flat)
        else:
            conv10 = None
        
     
        if branch_1d:
            if self.manual_seq_maxlength != None:
                input_seqlength = self.manual_seq_maxlength
            else:
                input_seqlength = self.seq_maxlength
        
            print ("input_seqlength=", input_seqlength)
        
            seq_input = Input(shape=(input_seqlength, 22), dtype='float32', name='seq_input')
            print ("seqinput", seq_input.shape)
            inputs.append(seq_input)

            seq_conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_input)
            seq_conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv1)
            seq_pool1 = MaxPooling1D(pool_size=2)(seq_conv1)
            seq_conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_pool1)
            seq_conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv2)
            seq_pool2 = MaxPooling1D(pool_size=2)(seq_conv2)
            seq_conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_pool2)
            seq_conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv3)
            seq_pool3 = MaxPooling1D(pool_size=2)(seq_conv3)
            seq_conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_pool3)
            seq_conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv4)
            seq_drop4 = Dropout(self.dropout)(seq_conv4)
            seq_pool4 = MaxPooling1D(pool_size=2)(seq_drop4)
    
            seq_conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_pool4)
            seq_conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv5)
            seq_drop5 = Dropout(0.5)(seq_conv5)
    
            seq_up6 = Conv1D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(seq_drop5))
            seq_merge6 = concatenate([seq_drop4,seq_up6], axis = 2)
            seq_conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_merge6)
            seq_conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv6)
    
            seq_up7 = Conv1D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(seq_conv6))
            seq_merge7 = concatenate([seq_conv3,seq_up7], axis = 2)
            seq_conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_merge7)
            seq_conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv7)
    
            seq_up8 = Conv1D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(seq_conv7))
            seq_merge8 = concatenate([seq_conv2,seq_up8], axis = 2)
            seq_conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_merge8)
            seq_conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv8)
    
            seq_up9 = Conv1D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(seq_conv8))
            seq_merge9 = concatenate([seq_conv1,seq_up9], axis = 2)
            seq_conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_merge9)
            seq_conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv9)
            seq_conv9 = Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv9)
            seq_conv10 = Conv1D(22, 1, activation='linear'   )(seq_conv9)
        

            #branch to classification
            seq_pool_down1 = AveragePooling1D(pool_size=2)(seq_drop5) # 7
            seq_conv_down1 = BatchNormalization()( Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_pool_down1) )
            seq_conv_down2 = BatchNormalization()( Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(seq_conv_down1) )
            seq_drop_down1 = Dropout(self.dropout)(seq_conv_down2)
            
            

            if branch_2d:
                seq_final_pool = GlobalAveragePooling1D()(seq_drop_down1)
                final = concatenate([final_pool, seq_final_pool], axis=1)
                print("concatenate",final.shape)
            else:
                seq_final_pool = GlobalAveragePooling1D(name="final")(seq_drop_down1)
                print("seq_final_pool",seq_final_pool.shape)
                final = seq_final_pool
            
            seq_softmax_out = Activation(self.softMaxAxis(2), name="seq_dec")(seq_conv10)
        else:
            seq_softmax_out=None
            final = final_pool
        
        
        print ("model inputs:", inputs)
        
        outputs = self.add_outputs_to_model(final, unet_decoder=conv10, seqnet_decoder=seq_softmax_out)
        
        model = Model(inputs=inputs, outputs=outputs)   
        return model

    def unet3(self):
        input_size = (self.img_dim,self.img_dim,self.nchannels)
        inputs = Input(input_size, dtype='float32', name='main_input')
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
    
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        #merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        #merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        #merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation = 'linear', name="dc_decoder")(conv9)
        
        
        flat = Flatten()(pool4)

        x, added_inputs = self.add_fclayers_to_model(flat)
        
        inputs = [inputs] + added_inputs
        print ("model inputs:", inputs)
        
        outputs = self.add_outputs_to_model(x, unet_decoder=conv10)
        
        model = Model(inputs=inputs, outputs=outputs)        
        #model = Model(input = inputs, output = conv10)
    
        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #model.summary()
    def get_model(self):
        dec_str = ""
        seq_dec_str = ""
        if not self.load_model:
            if self.model == "lenet":
                my_model = self.lenet(self.X, self.nb_classes, batch_norm=self.batch_norm, act=self.act)
            elif self.model == "lenet2":
                my_model = self.lenet2(self.X, self.nb_classes, batch_norm=self.batch_norm, act=self.act)
            elif self.model == "lenet3":
                my_model = self.lenet3(self.X, self.nb_classes, batch_norm=self.batch_norm, act=self.act)
            elif self.model =="unet":
                my_model = self.unet()
            elif self.model =="seqnet":
                my_model = self.seqnet()
            elif self.model == "fullconv":
                my_model = self.get_fullconv_model(branch_2d=self.use_dc_decoder, branch_1d=self.use_seq_decoder)
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
            
        if self.use_dc_decoder:
            dec_str = "_dec%.2f" % (self.dc_dec_weight)
        
        if self.use_seq_decoder:
            seq_dec_str = "_seq%.2f" % (self.seq_dec_weight)
        
        self.modlabel = "%s_%s%s%s_pt%i_%sx%sx%i_%i_bt%i_ep%i_f%ix%i_fc%ix%i_cath%s_%iG" % (
                            self.generic_label, self.model,
                            dec_str,
                            seq_dec_str,
                            int(self.use_pretrained),
                            str(self.img_dim), str(self.img_dim), self.nchannels, self.X.shape[0],
                            self.batch_size, self.epochs, self.kernel_shape[0], self.kernel_shape[1],
                            self.fc1, self.fc2, "".join(list(map(str, self.label_columns))), self.ngpus
                        )
        print ("Summary string for model:")
        print (self.modlabel)
        plot_model(my_model, to_file=os.path.join(self.outdir, self.modlabel+'_model-diagram.png'),show_shapes=True, show_layer_names=True, rankdir='TB')#'LR'
        return my_model

    def plot_sample_images(self, X, labels):
        try:
            # show some random example images
            fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, squeeze=False)
            randi = [random.randint(0, X.shape[0] - 1) for i in range(4)]
            test_imgs = [np.squeeze(X[i]) if self.nchannels == 1 else X[i] for i in randi ]
            titles = [labels[i] for i in randi]
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
            plt.show()
        except IndexError as ie:
            print ("Fix me: indexing error for test images!  ", ie)

    def prepare_dataset(self, show_classnames=False):
        
        if self.img_size_bins == []:
            if self.h5_input == "":
                self.getTrainTestSetMulti()
            else:
                self.getTrainTestSetMulti_h5()
            
            if self.show_images: self.plot_sample_images(self.X, self.labels)
            
            
            if 11 in self.label_columns:
                index_11 = self.label_columns.index(11)
                print ("col 11 index: ",index_11)
                self.Ys[index_11] = list(map(int, self.Ys[index_11] ))
            
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
                
                if self.show_images: self.plot_sample_images(self.dX[s], self.dlabels[s])
                
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
                    print (self.seqdict[self.labels[i]])
                    print (seq_from_onehot(self.seqdata[i, :, :]))
                    break
                else:
                    print ("Label not found in seqdict:",self.labels[i])

    def get_output_labels(self):
        my_outputs = []
        my_losses = {}
        my_loss_weights = {}
        my_metrics = {}
        # attach four new output layers instead of removed output layer
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
        
    def train(self, generic_label="test", load_model=False, show_classnames=False):
        if self.model == "none":
            print ("No model selected. Aborting.")
            return
        
        # get dataset
        self.prepare_dataset(show_classnames=show_classnames)
        
        print ("Data set shape: %s" % str(self.X.shape))
        
        print ("CNN:")
      
        print (self.X.shape)
        print ("Training this model:")
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
                                           orig_model, 
                                            monitor='val_loss', verbose=0, 
                                            save_best_only=True, save_weights_only=False, 
                                            mode='auto', period=self.checkpoints)
            callbacks.append(chk)
        
        my_input_data = []

        if not( self.model=='fullconv' and not self.use_dc_decoder):
            print("Using 2D input data", self.use_dc_decoder, self.model)
            my_input_data.append(self.X)
        
        my_outputs, my_losses, my_loss_weights, my_metrics = self.get_output_labels()
        
        # for classification outputs:
        output_labels = {my_outputs[o]:self.Ymats[o] for o in range(len(my_outputs))}
        
        # for regression outputs:
        if 11 in self.label_columns:
            output_labels["LENout"] = self.Ys[self.label_columns.index(11)]
        
                         
        if self.use_dc_decoder:
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
            my_input_data.append(self.seqdata)     

        print (output_labels.keys())
        self.keras_model.compile(
                                 loss=my_losses,
                                 optimizer=opt,
                                 metrics=my_metrics,
                                 loss_weights=my_loss_weights
                                 )
        
        if self.img_dim != None:
            self.history = self.keras_model.fit(my_input_data, output_labels,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=self.valsplit,
                  shuffle=True,
                  verbose=self.train_verbose
                  , callbacks=callbacks
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
            orig_model.save_weights(self.model_weights_file)
            with open(self.model_arch_file, "w") as json:
                json.write(orig_model.to_json())
            
        return self.modlabel
            
    def RF_baseline(self, X, Ys, modlabel):
        for i in range(len(Ys)):
            print ("Random Forest baseline:", i)
            X_train, X_test, y_train, y_test = train_test_split(np.array([x.flatten() for x in X]), Ys[i], test_size=0.33, random_state=0)
            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_test.shape[0], 'test samples')
           
            clf = RandomForestClassifier(n_estimators=100)  # ,random_state=123)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            # y_pred_proba = clf.predict_proba(X_test)    
            print (confusion_matrix(y_test, y_pred))
            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = 2 * (pre * rec) / (pre + rec)
            print ("accuracy ACC=(TP+TN/(FP+FN+TP+TN)=1-ERR  :  %.3f" % acc)
            print ("Precision=", pre)
            print ("Recall=", rec)
            print ("F1=", f1)
            
            with open("%s_%i_randomforest.txt" % (modlabel, i), "w") as rff:
                rff.write("acc:%f, pre:%f, rec:%f, f1:%f" % (acc, pre, rec, f1))
            # print "recall TP/(FN+TP): %.3f" % recall_score(y_test, y_pred)  

    def visualize_image_data(self, show_plots=False, use_pixels=True, use_pfp=True, pfp_layername="fc_last", ndims=2, perp=100
        , early_exaggeration=4.0 
        , learning_rate=100, angle=0.5
        , n_iter=1000, rseed=123, pfp_length=512, marker_size=1, n_jobs=1, save_path="", do_cluster_analysis=False, clustering_method="kmeans", scores_dict={}):
        
        
        assert clustering_method in ["kmeans", "hdbscan"]

        files_written = []
        
        if use_pfp:
            from predictChembl import get_pfp_generator_model
            pfp_generator = get_pfp_generator_model(self.model_arch_file, self.model_weights_file, pfp_layername)
            X_pfp = pfp_generator.predict(self.X)
            
        
        print (self.X.shape)
        X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1] * self.X.shape[2] * self.X.shape[3]))
        print (X.shape)
        
        
        
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
            emb = tsne.fit_transform(X)
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
                domlen = self.Ys[self.label_columns.index(11)]
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
                domlen = self.Ys[self.label_columns.index(11)]
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
        return files_written

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
                            max_img_size=299,
                            dc_decoder=False
                            
                        )
    if args.tsne_images: clf.visualize_image_data()
    
    # MAIN 
    if args.model != "none":
        clf.train(generic_label=args.label, load_model=False, show_classnames=args.classnames)
        
        if clf.history != None:
            clf.plot_curves(metric="loss")
            clf.plot_curves(metric="acc")
        
    if False:
        sub = subprocess.Popen("nvidia-smi", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print (sub.stdout.read())
        print (sub.stderr.read())
    
