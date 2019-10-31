import ProtConv2D.keras_chembl_classifier_v1 as cs
import sys

ngpus = int(sys.argv[1])
results_dir = "/home-us/ts149092/results/chembl"

input_modlabel="img-seq-clf_fullconv_imgE_seqlE_LAB_Dr0.0_relu_pt1_256x256x3_20798_bt128_ep200_f3x3_fc0x128_clfCATH_4G_split0.50_lo0.032_vlo4.244"


clf = cs.ChEMBL_Classifier(root_path="/home-us/ts149092/data",  batch_size=128, epochs=200, early_stopping='val_loss',
generic_label="chembl-pretrained",classifier='dense',fc1=1024, fc2=512, dropout=0.5, valsplit=0.3,
use_img_encoder=True, use_seq_encoder=True, seq_enc_arch="lstm",
use_biovec=False, biovec_model="/home-us/ts149092/data/uniprot_swissprot_2019-03-25_protvec128-3.model", biovec_length=128, biovec_flatten=True,
generate_protvec_from_seqdict=False, input_fasta_file="uniprot_swissprot_2019-03-25.fasta",
verbose=True, train_verbose=True, img_dim=256, ngpus=ngpus, outdir=results_dir,
use_embedding=True, manual_seq_maxlength=1400,
mol_fp_length=2048, morgan_radius=3,
pre_shuffle_indices=True,
names={ "prot_ID":"best_match", 
        "mol_ID":"molregno", 
        "inputs":["main_input", "seq_input", "mol_input"],
        "clf_targets":[ "pchembl_class" ], 
        "reg_targets":[ "pchembl_value" ]  } #
,
model_arch_file="/home-us/ts149092/results/img_plus_seq_clf/%s_model_arch.json"%input_modlabel, 
model_weights_file="/home-us/ts149092/results/img_plus_seq_clf/%s_model_weights.hdf5"%input_modlabel
)


clf.import_chembl_dataset(img_h5_file="/home-us/ts149092/data/ChEMBL_HEAVY_256x256x3_n871.hdf5", 
                            target_file="/home-us/ts149092/data/chembl_targets.csv", 
                            cmpd_file="/home-us/ts149092/data/chembl_mol.csv", 
                            activity_file="/home-us/ts149092/data/chembl_act.csv")


#clf.train_chembl()
clf.kfold_train_chembl(n_splits=5, shuffle=True, mode="StratifiedKFold", random_state=None, target_column="pchembl_class")

clf.plot_curves(metrics=["loss","pchembl_class_loss","pchembl_value_loss"])#,"pchembl_value_loss"])
clf.plot_curves(metrics=["pchembl_class_acc"])
clf.plot_sample_images()
