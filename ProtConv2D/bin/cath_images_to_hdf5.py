# SAVE IMAGE DATASET TO HDF5
# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
import os
import sys

import tables

import ProtConv2D.keras_cath_classifier as cs

dim = int(sys.argv[1])

atom_selection = "HEAVY"
if len(sys.argv) == 3:

    atom_selection = sys.argv[2]

assert atom_selection in ["HEAVY", "CA"]

if atom_selection == "HEAVY":
    img_folder = "pngbp"
elif atom_selection == "CA":
    img_folder = "pngca"

nchannels = 3
columns = "2,3,4,5,6,7,8,9,10,11,12"
root_path = "/hpc/projects/ts149092/cath"

if nchannels == 3:
    suffix = "_rgb.png"
else:
    suffix = "_dist.png"


clf = cs.CATH_Classifier(
    root_path,
    image_folder=img_folder,
    img_dim=dim,
    batch_size=64,
    epochs=200,
    model="fullconv",
    data_labels_filename="cath-domain-list.txt",
    label_columns=columns,
    png_suffix=suffix,
    nchannels=nchannels,
    sample_size=None,
    selection_filename="cath-dataset-nonredundant-S40.txt",
    idlength=7,
    kernel_shape="3,3",
    dim_ordering="channels_last",
    valsplit=0.4,
    save_resized_images=False,
    outdir="results",
    verbose=False,
    h5_backend="tables",
    use_img_encoder=False,
    use_seq_encoder=True,
)

# clf = cs.CATH_Classifier(root_path,
#                         image_folder=img_folder,
#                         img_dim=dim,
#                         data_labels_filename="cath-domain-list.txt",
#                         label_columns=columns,
#                         png_suffix="_rgb.png",
#                         nchannels=nchannels,
#                         sample_size=None,
#                         selection_filename="cath-dataset-nonredundant-S40.txt",
#                         idlength=7,
#                         dim_ordering="channels_last",
#                         save_resized_images=False,
#                         outdir="results",
#                         verbose=True,
#                         img_size_bins=[],
#                         img_bin_batch_sizes=[],
#                         flipdia_img=False,
#                         inverse_img=False,
#                         CATH_Y_labels_interpretable=False

#                         )


clf.prepare_dataset()
print("not found:", clf.not_found)

saved_files = clf.export_cath_dataset(
    "/home-us/ts149092/data", atom_selection, export_images=True, export_metadata=True
)

print(saved_files)
# hdf5_path = os.path.join(root_path, "CATH_%s_%s_%s.hdf5"%(atom_selection, "-".join(map(str,clf.X.shape)),columns.replace(",","-") )  )
# print ("Writing %s"%hdf5_path)

# if dim!=0:

#     print( clf.X.shape )

#     img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
#     data_shape = list(clf.X.shape)
#     data_shape[0] = 0
#     print( data_shape )

#     hdf5_file = tables.open_file(hdf5_path, mode='w')
#     #X_storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)
#     hdf5_file.create_array(hdf5_file.root, 'img', clf.X)
#     hdf5_file.create_array(hdf5_file.root, 'cath_codes', clf.labels)
#     hdf5_file.create_array(hdf5_file.root, 'class_labels', clf.Ys)
#     hdf5_file.close()
# else:


#     for x in clf.dX:
#         print ( x, clf.dX[x].shape)

#         #hdf5_path = "CATH-nonred-dist-dn%s.hdf5"%str(x)
#         img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
#         data_shape = list(clf.dX[x].shape)
#         data_shape[0] = 0
#         print( data_shape)


#         hdf5_file = tables.open_file(hdf5_path, mode='w')
#         #X_storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)
#         hdf5_file.create_array(hdf5_file.root, 'img', clf.dX[x])
#         hdf5_file.create_array(hdf5_file.root, 'cath_codes', clf.dlabels[x])
#         hdf5_file.create_array(hdf5_file.root, 'class_labels', clf.dYs[x])
#         hdf5_file.close()


# test created file


clf2 = cs.CATH_Classifier(
    root_path,
    image_folder=img_folder,
    img_dim=dim,
    batch_size=64,
    epochs=200,
    model="fullconv",
    data_labels_filename="cath-domain-list.txt",
    label_columns=columns,
    png_suffix=suffix,
    nchannels=nchannels,
    sample_size=None,
    selection_filename="cath-dataset-nonredundant-S40.txt",
    idlength=7,
    kernel_shape="3,3",
    dim_ordering="channels_last",
    valsplit=0.4,
    save_resized_images=False,
    outdir="results",
    verbose=False,
    h5_backend="tables",
    use_img_encoder=False,
    use_seq_encoder=True,
)

clf2.import_cath_dataset(img_h5_file=saved_files[1], metadata_file=saved_files[0])

print("Loaded dataset shape:", clf2.X.shape)
print(clf2.Ydict.keys())
print(clf2.labels[:5])
