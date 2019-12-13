
import argparse
import collections
import glob
import math
import os
import random
import subprocess
import sys
import time
import warnings

import h5py
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    homogeneity_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import biovec
import cv2

import prody
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.activations import linear, relu, sigmoid, softmax
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
from keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    History,
    ModelCheckpoint,
    ProgbarLogger,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.engine.topology import get_source_inputs
from keras.initializers import glorot_normal, he_normal, lecun_normal, lecun_uniform
from keras.layers import (
    GRU,
    LSTM,
    Activation,
    AveragePooling1D,
    AveragePooling2D,
    Bidirectional,
    Conv1D,
    Conv2D,
    Conv3D,
    CuDNNGRU,
    CuDNNLSTM,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    Lambda,
    Masking,
    MaxPooling1D,
    MaxPooling2D,
    Permute,
    RepeatVector,
    Reshape,
    SpatialDropout1D,
    SpatialDropout2D,
    TimeDistributed,
    UpSampling1D,
    UpSampling2D,
    concatenate,
    merge,
    multiply,
)
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.embeddings import Embedding
from keras.layers.merge import (
    add,
    average,
    concatenate,
    dot,
    maximum,
    multiply,
    subtract,
)
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.losses import categorical_crossentropy
from keras.models import Model, clone_model, model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import (
    Tokenizer,
    hashing_trick,
    one_hot,
    text_to_word_sequence,
)
from keras.regularizers import l1, l2
from keras.utils import (
    Sequence,
    layer_utils,
    multi_gpu_model,
    np_utils,
    plot_model,
    to_categorical,
)
from keras.utils.data_utils import get_file

from ProtConv2D.utils.utils import print_progress
from ProtConv2D.models.cath_classifier import CATH_Classifier

mpl.use("Agg")
plt.rcParams["figure.figsize"] = (20, 20)
sns.set()










if __name__ == "__main__":

    # PARSE ARGUMENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="cath", help="")
    ap.add_argument("--domlistname", default="cath-domain-list.txt", help="")
    ap.add_argument(
        "--model",
        default="lenet",
        choices=[
            "none",
            "unet",
            "lenet",
            "lenet2",
            "lenet3",
            "vgg16",
            "vgg19",
            "inception",
            "xception",
            "resnet50",
            "inception_resnetv2",
            "mobilenet",
            "densenet121",
            "densenet169",
            "densenet201",
        ],
        help="none means do not perform classfication.",
    )
    ap.add_argument("--pngpath", default="png2", help="")
    ap.add_argument("--dim", type=int, default=-1, help="")
    ap.add_argument("--bs", type=int, default=16, help="")
    ap.add_argument("--ep", type=int, default=10, help="")
    ap.add_argument("--kernel_shape", default="3,3", help="")
    ap.add_argument("--label_columns", default="2,3,4,5", help="")
    ap.add_argument(
        "--selection_filename",
        default="cath-dataset-nonredundant-S40.txt",
        help="'None' for no selection (all domains used).",
    )
    ap.add_argument("--png_suffix", default="_rgb.png", help="")
    ap.add_argument("--nchannels", type=int, choices=[1, 3], default=3, help="")
    ap.add_argument("--samples", type=int, default=0, help="")
    ap.add_argument("--label", default="cath2rgb", help="")
    ap.add_argument("--tsne_images", action="store_true", help="")
    ap.add_argument(
        "--dimord",
        choices=["channels_first", "channels_last"],
        default="channels_last",
        help="",
    )
    ap.add_argument("--valsplit", type=float, default=0.33, help="")
    ap.add_argument("--save_resized", action="store_true", help="")
    ap.add_argument("--outdir", default="results", help="")
    ap.add_argument("--use_pretrained", action="store_true", help="")
    ap.add_argument("--early_stopping", default="none", help="")
    ap.add_argument("--tensorboard", action="store_true", help="")
    ap.add_argument(
        "--opt", choices=["sgd", "rmsprop", "adam"], default="adam", help=""
    )
    ap.add_argument("--less", action="store_true", help="")
    ap.add_argument("--noBN", action="store_true", help="")
    ap.add_argument(
        "--act", choices=["relu", "elu", "selu", "tanh", "sigmoid"], default="relu"
    )
    ap.add_argument(
        "--classnames", action="store_true", help="Print all class names to screen."
    )
    ap.add_argument(
        "--lr",
        type=np.float32,
        default=-1,
        help="Learning rate. -1 means default of specified optimizer.",
    )
    ap.add_argument("--dropout", type=np.float32, default=0.25, help="Dropout fraction")
    ap.add_argument(
        "--img_bins", default="none", help="Example: 16,32,64,128,256,512,1024"
    )
    ap.add_argument("--img_bins_bs", default="none", help="")
    ap.add_argument("--flipdia_img", action="store_true", help="")
    ap.add_argument("--inverse_img", action="store_true", help="")
    ap.add_argument("--model_arch_file", default=None, help="")
    ap.add_argument("--model_weights_file", default=None, help="")
    ap.add_argument("--show_images", action="store_true", help="")

    args = ap.parse_args()
    print("Settings:", args)

    clf = CATH_Classifier(
        args.path,
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
        img_size_bins=[]
        if args.img_bins in ["none", "None", "0", "[]"]
        else list(map(int, args.img_bins.split(","))),
        img_bin_batch_sizes=[]
        if args.img_bins_bs in ["none", "None", "0", "[]"]
        else list(map(int, args.img_bins_bs.split(","))),
        model_arch_file=args.model_arch_file,
        model_weights_file=args.model_weights_file,
        show_images=args.show_images,
        min_img_size=-1,
        max_img_size=299,
    )
    if args.tsne_images:
        clf.visualize_image_data()

    # MAIN
    if args.model != "none":
        clf.train(
            generic_label=args.label, load_model=False, show_classnames=args.classnames
        )

        if clf.history != None:
            clf.plot_curves(metrics=["loss"])
            # clf.plot_curves(metrics="acc")

    if False:
        sub = subprocess.Popen(
            "nvidia-smi", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(sub.stdout.read())
        print(sub.stderr.read())

