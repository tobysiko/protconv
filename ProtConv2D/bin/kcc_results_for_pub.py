import ast
import glob
import os

import matplotlib as mpl
import pandas as pd

import ProtConv2D.models.cath_classifier as cs
from predictChembl import add_protein_fingerprints, get_pfp_generator_model
from typing import Dict
mpl.use("Agg")


results_path = "results"
last_layer = "fc_last"
n_replicates = 5
root_path = "cath"
arch_names = ["densenet121", "vgg16", "resnet50", "inception"]


results_table = []


fc_lengths = ["128", "512"]
for i in range(n_replicates):
    print(i)
    for a in arch_names:
        print(a)
        if a in ["inception"]:
            dim = 299
        elif a in ["nasnetlarge"]:
            dim = 331
        else:
            dim = 224

        for l in fc_lengths:

            print(l)
            model_arch_file = glob.glob(
                os.path.join(
                    results_path,
                    "0%i" % (i + 1),
                    "kcc_%s*fc0x%s*_model_arch.json" % (a, l),
                )
            )[0]
            model_weights_file = glob.glob(
                os.path.join(
                    results_path,
                    "0%i" % (i + 1),
                    "kcc_%s*fc0x%s*_model_weights.hdf5" % (a, l),
                )
            )[0]
            model_history = glob.glob(
                os.path.join(
                    results_path,
                    "0%i" % (i + 1),
                    "kcc_%s*fc0x%s*_model_history.json" % (a, l),
                )
            )[0]
            base = os.path.basename(model_arch_file)[:-16]
            print(base)
            with open(model_history) as hist:
                training_history = ast.literal_eval(hist.read())

            # print( training_history.keys() )
            # ['loss', 'val_Tout_loss', 'val_Aout_acc', 'val_Aout_loss', 'val_Hout_loss', 'Cout_loss', 'val_Tout_acc', 'val_Cout_acc', 'Aout_acc', 'Aout_loss', 'Hout_loss', 'lr', 'Tout_acc', 'Hout_acc', 'Cout_acc', 'val_Cout_loss', 'val_loss', 'val_Hout_acc', 'Tout_loss']

            n_epochs = len(training_history["loss"])

            train_loss = training_history["loss"][-1]
            val_loss = training_history["val_loss"][-1]

            train_C_acc = training_history["Cout_acc"][-1]
            val_C_acc = training_history["val_Cout_acc"][-1]

            train_A_acc = training_history["Aout_acc"][-1]
            val_A_acc = training_history["val_Aout_acc"][-1]

            train_T_acc = training_history["Tout_acc"][-1]
            val_T_acc = training_history["val_Tout_acc"][-1]

            train_H_acc = training_history["Hout_acc"][-1]
            val_H_acc = training_history["val_Hout_acc"][-1]

            # load Keras model
            # pfp_generator = get_pfp_generator_model(model_arch_file, model_weights_file, last_layer)

            fname = "CATH_20798-%i-%i-3.hdf5" % (dim, dim)

            hname = os.path.join(root_path, fname)

            # if model == "unet": dim=256

            clf = cs.CATH_Classifier(
                root_path,
                image_folder="pngbp",
                img_dim=dim,
                batch_size=64,
                epochs=200,
                model=a,
                data_labels_filename="cath-domain-list.txt",
                label_columns="2,3,4,5",
                png_suffix="_rgb.png",
                nchannels=3,
                sample_size=None,
                selection_filename="cath-dataset-nonredundant-S40.txt",
                idlength=7,
                kernel_shape="3,3",
                dim_ordering="channels_last",
                valsplit=0.4,
                save_resized_images=False,
                outdir=results_path,
                use_pretrained=True,
                early_stopping="val_loss",
                tensorboard=False,
                optimizer="adam",
                verbose=False,
                batch_norm=True,
                act="relu",
                learning_rate=-1,
                img_size_bins=[],
                dropout=0.25,
                img_bin_batch_sizes=[],
                flipdia_img=False,
                inverse_img=False,
                fc1=0,
                fc2=int(l),
                keras_top=True,
                ngpus=1,
                train_verbose=False,
                generic_label="kcc",
                h5_input=hname,
                h5_backend="h5py",
                #dc_decoder=False,
                model_arch_file=model_arch_file,
                model_weights_file=model_weights_file,
            )

            clf.prepare_dataset(show_classnames=False)
            # print (clf.keras_model)
            clf.keras_model = clf.get_model()
            # print (clf.keras_model)
            # print (clf.keras_model.summary())

            scores:Dict = {}
            new_files = clf.visualize_image_data(
                use_pixels=False,
                use_pfp=True,
                save_path=os.path.join(results_path, "0%i" % (i + 1)),
                do_cluster_analysis=True,
                clustering_method="kmeans",
                scores_dict=scores,
            )
            print(
                i,
                a,
                l,
                n_epochs,
                train_loss,
                val_loss,
                train_C_acc,
                val_C_acc,
                train_A_acc,
                val_A_acc,
                train_T_acc,
                val_T_acc,
                train_H_acc,
                val_H_acc,
                scores["pfprints_Class_homogeneity_score"],
                scores["pfprints_Architecture_homogeneity_score"],
                scores["pfprints_Topology_homogeneity_score"],
                scores["pfprints_Homology_homogeneity_score"],
            )
            results_table.append(
                [
                    a,
                    l,
                    n_epochs,
                    train_loss,
                    val_loss,
                    train_C_acc,
                    val_C_acc,
                    train_A_acc,
                    val_A_acc,
                    train_T_acc,
                    val_T_acc,
                    train_H_acc,
                    val_H_acc,
                    scores["pfprints_Class_homogeneity_score"],
                    scores["pfprints_Architecture_homogeneity_score"],
                    scores["pfprints_Topology_homogeneity_score"],
                    scores["pfprints_Homology_homogeneity_score"],
                ]
            )

            # print( new_files )

df = pd.DataFrame(
    results_table,
    columns=[
        "arch",
        "pfp_length",
        "n_epochs",
        "train_loss",
        "val_loss",
        "train_C_acc",
        "val_C_acc",
        "train_A_acc",
        "val_A_acc",
        "train_T_acc",
        "val_T_acc",
        "train_H_acc",
        "val_H_acc",
        "pfprints_Class_homogeneity_score",
        "pfprints_Architecture_homogeneity_score",
        "pfprints_Topology_homogeneity_score",
        "pfprints_Homology_homogeneity_score",
    ],
)
df.to_csv(os.path.join(results_path, "kcc_results_for_pub__summary_results.csv"))

