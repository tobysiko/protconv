import os
import sys

# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
import matplotlib as mpl

import ProtConv2D.keras_cath_classifier_v3 as cs

mpl.use("Agg")

model_path = os.path.abspath(sys.argv[1])


root_path = os.path.abspath("cath")
ngpus = 4
results_dir = os.path.abspath("results")

if ngpus == 8:
    bs = 512
elif ngpus == 4:
    bs = 48
else:
    bs = 32
# bs=64*ngpus

dim = 256
nchannels = 3

clf = cs.CATH_Classifier(
    root_path, outdir=results_dir, img_dim=dim, nchannels=nchannels
)

clf.load_model_from_files(model_path)

clf.import_cath_dataset(
    img_h5_file=os.path.join(
        root_path, "CATH_HEAVY_%ix%ix%i_n20798.hdf5" % (dim, dim, nchannels)
    ),
    metadata_file=os.path.join(root_path, "CATH_metadata.csv"),
)


try:
    clf.load_training_history(model_path + "_results-history.csv")

    clf.plot_curves(metrics=["loss"])
except Exception as e:
    print(e)


try:
    clf.plot_curves(
        metrics=[
            "cath01_class_acc",
            "cath02_architecture_acc",
            "cath03_topology_acc",
            "cath04_homologous_superfamily_acc",
        ]
    )
except Exception as e:
    print("ERROR IN plot_curves", e)

try:
    clf.plot_curves(metrics=["sequence_length_loss"])
except Exception as e:
    print(e)

try:
    if "seq2img" in clf.modlabel:
        clf.plot_reconstruction_samples_seq2img()
    elif "fullconv" in clf.modlabel:
        clf.plot_reconstruction_samples_unet()
except Exception as e:
    print("ERROR IN plot_reconstruction_samples_seq2img", e)

try:
    clf.plot_sample_images()
except Exception as e:
    print("ERROR IN plot_sample_images", e)

try:
    new_files = clf.visualize_cath_image_data(
        show_plots=False,
        use_pixels=False,
        use_pfp=True,
        pfp_layername=[
            "fc_last",
            "final",
            "global_average_pooling2d_1",
            "img_conv_bottleneck",
            "dropout_2",
        ],  # list of potential layers to extract fingerprints from. pick first one in list that exists.
        ndims=2,
        perp=100,
        early_exaggeration=4.0,
        learning_rate=100,
        angle=0.5,
        n_iter=1000,
        rseed=123,
        marker_size=1,
        n_jobs=1,
        save_path=results_dir,
        do_cluster_analysis=False,
        clustering_method="kmeans",
        scores_dict={},
    )
except Exception as e:
    print("ERROR IN visualize_cath_image_data", e)
