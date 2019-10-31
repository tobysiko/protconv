# SAVE IMAGE DATASET TO HDF5
#http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
import tables, os, sys, glob, collections
import ProtConv2D.keras_chembl_classifier_v1 as cs
import pandas as pd
dim = int(sys.argv[1])

atom_selection = "HEAVY"
#if len(sys.argv)==3:

#    atom_selection = sys.argv[2]

assert atom_selection in ["HEAVY","CA"]

if atom_selection=="HEAVY":
    img_folder="pngbp"
elif atom_selection=="CA":
    img_folder="pngca"

nchannels = 3
columns = "2,3,4,5,6,7,8,9,10,11,12"
root_path = "/hpc/projects/ts149092/chembl"

if nchannels==3:
    suffix = "_rgb.png"
else:
    suffix = "_dist.png"

if True:
    pdb_id_list = [os.path.basename(f).strip("_rgb.png") for f in glob.glob( os.path.join( root_path, "png", "*rgb.png"  )  )]
    with open( os.path.join("/home-us/ts149092/data", "chembl_pdb_id_list.txt"), 'w'  ) as f:
        f.write("\n".join(pdb_id_list))



clf = cs.ChEMBL_Classifier(
                root_path, image_folder="png", 
                img_dim=dim, batch_size=64, epochs=200, 
                model="fullconv", data_labels_filename="chembl_pdb_id_list.txt", 
                label_columns=columns, 
                png_suffix=suffix, nchannels=nchannels, 
                sample_size=None, selection_filename=None, 
                idlength=6, kernel_shape="3,3",dim_ordering="channels_last", 
                valsplit=0.4, save_resized_images=False, outdir="results", 
                pdb_folder="chains",
                verbose=False,
                h5_backend="tables",
                dataset = 'chembl',
                use_img_encoder=False,
                seq_from_pdb=True,
                use_seq_encoder=True,
                )
if True:
    files = [f for f in glob.glob( os.path.join( root_path, "png", "*_rgb.png"  )  )]
    clf.read_images(files)

    print (clf.X.shape)
    print (len(clf.labels))

    print (clf.not_found)

    clf.get_sequence_data()
    print(len(clf.sequences))

    expath = "/home-us/ts149092/data"
    clf.export_chembl_dataset(expath, atom_selection)

    chembl_df = pd.read_csv("/home-us/ts149092/data/chembl_act_traintest_set.csv")
    print(chembl_df.info())

    unique_mols = chembl_df.drop_duplicates(subset=['molregno'])  
    print(unique_mols.info())  
    unique_mols.to_csv("/home-us/ts149092/data/chembl_mol.csv", columns=["molregno","comp_pref_name","comp_chembl_id","comp_class_id","molecule_type","standard_inchi","mw_freebase","alogp","aromatic_rings","canonical_smiles"])

    target_df = pd.read_csv("/home-us/ts149092/data/chembl_act_traintest_set.csv")
    unique_targets = target_df.drop_duplicates(subset=['best_match'])
    unique_targets.to_csv("/home-us/ts149092/data/chembl_targets.csv", columns=["target_chembl_id", "target_type", "target_pref_name", "target_chembl_id", 
                                                                        "component_id", "targcomp_id", "homologue", "protein_class_id", "pref_name", "protein_class_desc",
                                                                        "class_level", "domain_id", "domain_start_position", "domain_end_position", "domain_type", "source_domain_id", 
                                                                        "domain_name", "sequence_component_type", "sequence", "variant_sequence", 
                                                                        "variant_mutation", "variant_organism", "protein_familiy_level1", "protein_familiy_level2", 
                                                                        "protein_familiy_level3", "protein_familiy_level4", "protein_familiy_level5", "protein_familiy_level6", 
                                                                        "uniprot_accession", "mapped_chembl_id", "target_description", 
                                                                        "component_type", "chembl_component_id", "frac_match", "best_match"])


    act_df = pd.read_csv("/home-us/ts149092/data/chembl_act_traintest_set.csv")
    print(act_df.info())
    act_df.to_csv("/home-us/ts149092/data/chembl_act.csv", columns=["activity_id", "activity_comment", "pchembl_value", "pchembl_class", "molregno", "best_match"]) 

clf.import_chembl_dataset(img_h5_file="/home-us/ts149092/data/ChEMBL_HEAVY_256x256x3_n871.hdf5",target_file="/home-us/ts149092/data/chembl_targets.csv", 
                        cmpd_file="/home-us/ts149092/data/chembl_mol.csv", activity_file="/home-us/ts149092/data/chembl_act.csv")

assert False
#clf.import_chembl_dataset(img_h5_file=None, metadata_file=os.path.join( root_path, "chembl_act_traintest_set.csv"  ) )

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
print("not found:",clf.not_found )

saved_files = clf.export_cath_dataset("/home-us/ts149092/data", atom_selection, export_images=True, export_metadata=True)

print (saved_files)
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
                root_path, image_folder=img_folder, 
                img_dim=dim, batch_size=64, epochs=200, 
                model="fullconv", data_labels_filename="cath-domain-list.txt", 
                label_columns=columns, 
                png_suffix=suffix, nchannels=nchannels, 
                sample_size=None, selection_filename="cath-dataset-nonredundant-S40.txt", 
                idlength=7, kernel_shape="3,3",dim_ordering="channels_last", 
                valsplit=0.4, save_resized_images=False, outdir="results", 
                verbose=False,

                h5_backend="tables",
                
                use_img_encoder=False,
                
                use_seq_encoder=True,
                )

clf2.import_cath_dataset(img_h5_file=saved_files[1], metadata_file=saved_files[0])

print ("Loaded dataset shape:",clf2.X.shape)
print (clf2.Ydict.keys())
print (clf2.labels[:5])