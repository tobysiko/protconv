
# coding: utf-8

# # Prepare ChEMBL data set for training machine learning predictor of pChEMBL activity value (and activity class)

# IMports


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#Plotting parameter defaults
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=False)
plt.rcParams['figure.figsize']=(16,16)
import math

import pyodbc


def list_columns(obj, cols=4, columnwise=True, gap=4):
    """
    Print the given list in evenly-spaced columns.

    Parameters
    ----------
    obj : list
        The list to be printed.
    cols : int
        The number of columns in which the list should be printed.
    columnwise : bool, default=True
        If True, the items in the list will be printed column-wise.
        If False the items in the list will be printed row-wise.
    gap : int
        The number of spaces that should separate the longest column
        item/s from the next column. This is the effective spacing
        between columns based on the maximum len() of the list items.
    """

    sobj = [str(item) for item in obj]
    if cols > len(sobj): cols = len(sobj)
    max_len = max([len(item) for item in sobj])
    if columnwise: cols = int(math.ceil(float(len(sobj)) / float(cols)))
    plist = [sobj[i: i+cols] for i in range(0, len(sobj), cols)]
    if columnwise:
        if not len(plist[-1]) == cols:
            plist[-1].extend(['']*(len(sobj) - len(plist[-1])))
        plist = zip(*plist)
    printer = '\n'.join([
        ''.join([c.ljust(max_len + gap) for c in p])
        for p in plist])
    print (printer)


    
    
    
conn = pyodbc.connect("DSN=impaladsn", autocommit=True) 
pd.read_sql("show databases;", conn)



list_columns(pd.read_sql("show tables in assay_chembl_use;", conn).iloc[:,0].values, cols=3)


#See ChEMBL schema documentation here:
#ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_24_1/schema_documentation.txt

if False: # THIS EXCEEDED 50GB RAM LIMIT
  act_file = "chembl241_pchembl_activities_single_table.csv"
  print (act_file,"exists? ",os.path.exists(act_file))
  if os.path.exists(act_file):
    activities = pd.read_csv(act_file)
  else:
    activities = pd.read_sql("""SELECT act.activity_comment,
        act.activity_id,
        act.assay_id,
        act.record_id,
        act.molregno,
        act.pchembl_value,
        act.standard_value,
        act.standard_relation,
        act.standard_type,
        act.standard_units,
        act.data_validity_comment,
        ass.tid,
        ass.cell_id,
        ass.tissue_id,
        ass.variant_id,
        ass.confidence_score,
        ass.assay_category,
        ass.assay_type,
        mol.pref_name AS comp_pref_name,
        mol.chembl_id AS comp_chembl_id,
        mol.molecule_type,
        str.canonical_smiles,
        com.mw_freebase,
        com.hba,
        com.hbd,
        com.psa,
        com.rtb,
        com.ro3_pass,
        com.num_ro5_violations,
        com.acd_most_apka,
        com.acd_most_bpka,
        com.acd_logp,
        com.acd_logd,
        com.molecular_species,
        com.heavy_atoms,
        com.qed_weighted,
        com.alogp,
        com.aromatic_rings,        
        td.target_type,
        td.pref_name AS target_pref_name,
        td.tax_id,
        td.chembl_id AS target_chembl_id,
        tc.component_id,
        tc.targcomp_id,
        tc.homologue,
        cc.protein_class_id,
        cc.comp_class_id,
        pc.pref_name AS protein_class_pref_name,
        pc.short_name AS protein_class_short_name,
        pc.protein_class_desc,
        pc.class_level,
        cd.domain_id,
        cd.start_position AS domain_start_position,
        cd.end_position AS domain_end_position,
        dom.domain_type,
        dom.source_domain_id,
        dom.domain_name,
        dom.domain_description,
        seq.component_type AS sequence_component_type,
        seq.sequence,
        var.sequence AS variant_sequence,
        var.mutation AS variant_mutation,
        var.organism AS variant_organism,
        fam.l1 AS protein_family_level1,
        fam.l2 AS protein_family_level2,
        fam.l3 AS protein_family_level3,
        fam.l4 AS protein_family_level4,
        fam.l5 AS protein_family_level5,
        fam.l6 AS protein_family_level6
        
    FROM assay_chembl_use.activities_chembl_v24_1 AS act 
    LEFT JOIN assay_chembl_use.assays_chembl_v24_1 AS ass ON act.assay_id = ass.assay_id
    LEFT JOIN assay_chembl_use.molecule_dictionary_chembl_v24_1 AS mol ON act.molregno = mol.molregno
    LEFT JOIN assay_chembl_use.compound_properties_chembl_v24_1 AS com ON act.molregno = com.molregno
    LEFT JOIN assay_chembl_use.compound_structures_chembl_v24_1 AS str ON act.molregno = str.molregno
    LEFT JOIN assay_chembl_use.target_dictionary_chembl_v24_1 AS td ON ass.tid = td.tid
    LEFT JOIN assay_chembl_use.target_components_chembl_v24_1 AS tc ON ass.tid = tc.tid
    LEFT JOIN assay_chembl_use.component_class_chembl_v24_1 AS cc ON tc.component_id = cc.component_id
    LEFT JOIN assay_chembl_use.protein_classification_chembl_v24_1 AS pc ON cc.protein_class_id = pc.protein_class_id
    LEFT JOIN assay_chembl_use.component_domains_chembl_v24_1 AS cd ON tc.component_id = cd.component_id
    LEFT JOIN assay_chembl_use.domains_chembl_v24_1 AS dom ON cd.domain_id = dom.domain_id
    LEFT JOIN assay_chembl_use.component_sequences_chembl_v24_1 AS seq ON tc.component_id = seq.component_id
    LEFT JOIN assay_chembl_use.variant_sequences_chembl_v24_1 AS var ON ass.variant_id = var.variant_id
    LEFT JOIN assay_chembl_use.protein_family_classification_chembl_v24_1 AS fam ON cc.protein_class_id = fam.protein_class_id
    WHERE act.pchembl_value IS NOT NULL AND td.target_type = 'SINGLE PROTEIN' AND ass.confidence_score = 9
    """, conn)
    print (len(activities["molregno"].unique()))
    list_columns(activities.columns.values,cols=3)
    print( activities.info())
    activities.to_csv(act_file, encoding="utf8")

    
    
act_file = "chembl241_pchembl_single_protein_activities.csv"
print (act_file,"exists? ",os.path.exists(act_file))
if os.path.exists(act_file):
    activities = pd.read_csv(act_file)
else:
    activities = pd.read_sql("""SELECT act.activity_comment,
        act.activity_id,
        act.assay_id,
        act.record_id,
        act.molregno,
        act.pchembl_value,
        act.standard_value,
        act.standard_relation,
        act.standard_type,
        act.standard_units,
        act.data_validity_comment,
        ass.tid,
        ass.confidence_score,
        td.target_type
        
    FROM assay_chembl_use.activities_chembl_v24_1 AS act 
    LEFT JOIN assay_chembl_use.assays_chembl_v24_1 AS ass ON act.assay_id = ass.assay_id
    LEFT JOIN assay_chembl_use.target_dictionary_chembl_v24_1 AS td ON ass.tid = td.tid
    WHERE act.pchembl_value IS NOT NULL AND td.target_type = 'SINGLE PROTEIN' AND ass.confidence_score = 9
    """, conn)
    print (len(activities["activity_id"].unique()))
    list_columns(activities.columns.values,cols=3)
    print( activities.info())
    activities.to_csv(act_file, encoding="utf8")
activities=activities[ [c for c in activities.columns if not 'Unnamed' in c] ] # remove columns resulting from saving index to CSV
    
    
    
    
    
target_list = activities["tid"].unique()
print(len(target_list))
tar_file = "chembl241_pchembl_single_protein_targets.csv"
print (tar_file,"exists? ",os.path.exists(tar_file))
if os.path.exists(tar_file):
	targets = pd.read_csv(tar_file)
else:
	query="""SELECT 
      td.tid,
			td.tax_id,
			td.organism,
			td.target_type,
			td.pref_name AS target_pref_name,
			td.chembl_id AS target_chembl_id,
			tc.component_id,
			tc.targcomp_id,
			tc.homologue,
			cc.protein_class_id,
			cc.comp_class_id,
			pc.pref_name AS protein_class_pref_name,
			pc.short_name AS protein_class_short_name,
			pc.protein_class_desc,
			pc.class_level,
			seq.component_type AS sequence_component_type,
			seq.sequence,
			fam.l1 AS protein_family_level1,
			fam.l2 AS protein_family_level2,
			fam.l3 AS protein_family_level3,
			fam.l4 AS protein_family_level4,
			fam.l5 AS protein_family_level5,
			fam.l6 AS protein_family_level6

	FROM assay_chembl_use.target_dictionary_chembl_v24_1 AS td 
	LEFT JOIN assay_chembl_use.target_components_chembl_v24_1 AS tc ON td.tid = tc.tid
	LEFT JOIN assay_chembl_use.component_class_chembl_v24_1 AS cc ON tc.component_id = cc.component_id
	LEFT JOIN assay_chembl_use.protein_classification_chembl_v24_1 AS pc ON cc.protein_class_id = pc.protein_class_id

	LEFT JOIN assay_chembl_use.component_sequences_chembl_v24_1 AS seq ON tc.component_id = seq.component_id

	LEFT JOIN assay_chembl_use.protein_family_classification_chembl_v24_1 AS fam ON cc.protein_class_id = fam.protein_class_id
 	 WHERE td.tid IN (%s);
  """%( ", ".join( ['%s'%str(t)  for t in target_list ]   ))
  #	LEFT JOIN assay_chembl_use.component_domains_chembl_v24_1 AS cd ON tc.component_id = cd.component_id
	#		LEFT JOIN assay_chembl_use.domains_chembl_v24_1 AS dom ON cd.domain_id = dom.domain_id
	print(query)
	targets = pd.read_sql(query, conn)
	print (len(targets["target_chembl_id"].unique()))
	list_columns(targets.columns.values,cols=3)
	
	chembl_up = pd.read_table("chembl_uniprot_mapping.txt", header=None, skiprows=1, names=["uniprot_accession", "mapped_chembl_id", "target_description", "component_type"], delimiter="\t")

	print (len( chembl_up["uniprot_accession"].unique() ), "unique uniprot ids")
	print (len( chembl_up["mapped_chembl_id"].unique() ), "unique chembl ids")
	chembl_up

	tar_file = "chembl241_pchembl_single_protein_targets_uniprot.csv"
	targets = targets.merge(chembl_up.loc[:,["mapped_chembl_id","uniprot_accession","target_description"]], how="left", left_on="target_chembl_id", right_on="mapped_chembl_id")

	
	print( targets.info())
	targets.to_csv(tar_file, encoding="utf8")
targets=targets[ [c for c in targets.columns if not 'Unnamed' in c] ] # remove columns resulting from saving index to CSV



mol_list = activities["molregno"].unique()
print(len(mol_list))
mol_file = "chembl241_pchembl_single_protein_molecules.csv"
print (mol_file,"exists? ",os.path.exists(mol_file))
if os.path.exists(mol_file):
  molecules = pd.read_csv(mol_file)
else:
	molecule_dfs=[] 
  #can only query for 10000 molecules at a time
	for i in range( int(len(mol_list)/10000)):
		print ("query %i of %i"%(i+1, int(len(mol_list)/10000)))
		print ( i*9999, min((i+1)*9999, len(mol_list))   )
		sub_list = ['%s'%str(m)  for m in mol_list[i*9999:min((i+1)*9999, len(mol_list))] ]		
		print (len(sub_list), sub_list[:5])
		query ="""SELECT mol.molregno,
    mol.pref_name AS comp_pref_name,
			mol.chembl_id AS comp_chembl_id,
			mol.molecule_type,
			str.standard_inchi,
			str.canonical_smiles,
			com.mw_freebase,
			com.hba,
			com.hbd,
			com.psa,
			com.rtb,
			com.ro3_pass,
			com.num_ro5_violations,
			com.acd_most_apka,
			com.acd_most_bpka,
			com.acd_logp,
			com.acd_logd,
			com.molecular_species,
			com.heavy_atoms,
			com.qed_weighted,
			com.alogp,
			com.aromatic_rings  

	FROM assay_chembl_use.molecule_dictionary_chembl_v24_1 AS mol 
LEFT JOIN assay_chembl_use.compound_properties_chembl_v24_1 AS com ON mol.molregno = com.molregno
LEFT JOIN assay_chembl_use.compound_structures_chembl_v24_1 AS str ON mol.molregno = str.molregno
  WHERE mol.molregno in (%s)
  """%( ", ".join( sub_list   )) 
		df = pd.read_sql(query , conn, index_col='molregno')
		print(df.info())
		molecule_dfs.append( df )
		print(len(molecule_dfs))
	molecules = pd.concat( molecule_dfs)
	#molecules = molecules.join(molecule_dfs[1:])
	print(molecules.inlinedex.duplicated(keep='first').sum() )
	#print (len(molecules["molregno"].unique()))
	#list_columns(molecules.columns.values,cols=3)

	molecules.to_csv(mol_file, encoding="utf8")
print( molecules.info())
molecules=molecules[ [c for c in molecules.columns if not 'Unnamed' in c] ] # remove columns resulting from saving index to CSV



#activities


# map chembl_id to uniprot from official Chembl23 file at:
# ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_23




## In[13]:
#
#
#activities.info()
#
#
## In[14]:
#
#
#print len(activities["uniprot_accession"].unique())
#
#
## In[15]:
#
#
#del activities["mapped_chembl_id"]
#
#
## In[16]:
#
#
struc_df = pd.read_excel("human_proteome_with_structures.xlsx", sheetname="tabularResults")

#convert joint sequence & SS column into two seperate columns
struc_df["seq"] = struc_df["Sequence and Secondary Structure"].apply(lambda x: x.split('#')[0].strip())
struc_df["sec_struc"] = struc_df["Sequence and Secondary Structure"].apply(lambda x: x.split('#')[1].strip())
del struc_df["Sequence and Secondary Structure"]

print (struc_df.shape)
#
#
## In[17]:
#
#
chains_df = pd.read_excel("human_proteome_chains_with_start_stop.xlsm", sheetname=1)
print (chains_df.shape)
chains_df.columns.values[0] = "DB ID"
chains_df.columns.values[1] = "PDB ID"
chains_df.columns.values[2] = "Chain ID"
chains_df
#
#
## In[18]:
#
#
# remove unwanted PDB structures
bad_pdb_list = ["5GJR"]
struc_df = struc_df[~struc_df["PDB ID"].isin(bad_pdb_list)]

#
## In[19]:
#
#
chains_df["pdb_chain"] = ["%s_%s"%(a,b) for a,b in zip(chains_df.iloc[:,1].values, chains_df.iloc[:,2].values) ]

merged_df = struc_df.merge(chains_df, on=None)


# In[20]:


print( merged_df.shape)
list_columns(merged_df)

#merged_df
#
#
## In[21]:
#
#
pdb2up = merged_df.loc[:,["pdb_chain", "DB ID", "seq"]]

print (pdb2up.shape)
pdb2up = pdb2up.drop_duplicates()
print (pdb2up.shape)
assert pdb2up.shape[0] == len(pdb2up["pdb_chain"].unique())

#
## In[22]:
#
#
target2upseq = targets.loc[:,["uniprot_accession", "sequence",  "component_id"]]
print (len(targets["sequence"].unique()), "unique sequences")
print (len(targets["uniprot_accession"].unique()), "unique uniprot accessions")
#
#
print (target2upseq.shape)
#target2upseq = target2upseq.drop_duplicates()
#print target2upseq.shape
#
#
## In[23]:
#
#
merged_df[ merged_df["seq"]!=np.nan]
#
#
## In[24]:
#
#



def mapUP2PDB(up, chseq, pdb_df, sufficient_score=50):
    
    chains = pdb_df[pdb_df["DB ID"]==up].loc[:,["pdb_chain","seq"]]
    
    #chains = chains.drop_duplicates(subset="seq")
    if chains.empty: 
        return "NO_MATCH",np.nan # TODO: consider homology model!
    else:
        #chains = chains[chains["seq"]==chseq]
        # DO PAIRWISE SEQUENCE ALIGNMENT!
        chains["aln_scores"] = chains["seq"].apply(lambda x: pairwise2.align.globalxx(chseq, x, score_only=True) )
       
        chains = chains.sort_values(by="aln_scores", ascending=False)
        maxscore = chains.iloc[0,-1]
        if maxscore < sufficient_score:
            return "INSUFFICIENT_MATCH",np.nan
        
        
        return list(chains[chains["aln_scores"]==maxscore]["pdb_chain"].values), maxscore

if not os.path.exists("chembl241_target_uniprot_to_pdb_chain.csv"):
	#http://biopython.org/DIST/docs/api/Bio.pairwise2-module.html
	from Bio import pairwise2
	from MLhelpers import print_progress
	
	col_comp=[]
	col_var=[]
	col_up = []
	col_score = []
	col_upseq = []
	col_frac = []
	col_chains = []
	col_best = []
	for i in xrange(target2upseq.shape[0]):
			ix = target2upseq.index[i]
			print_progress(i,target2upseq.shape[0])
			up = target2upseq.loc[ix,"uniprot_accession"]
			comp_id= target2upseq.loc[ix,"component_id"]
			var_id= target2upseq.loc[ix,"variant_id"]
			upvar =  target2upseq.loc[ix,"variant_sequence"]

			if str(upvar).lower() not in ["nan","none"]:

					upseq = upvar
			else:
					upseq = target2upseq.loc[ix,"sequence"]

			chain_matches, score = mapUP2PDB(up, upseq, merged_df )

			best_match = chain_matches
			if chain_matches != [] and not chain_matches in ["NO_MATCH","INSUFFICIENT_MATCH"]:
					# chose between multiple matches from alignment:
					# 1) prefer lower (=better) resolution value
					# 2) prefer smaller ligand size, because the structure will be less biased towards that ligand and more generic
					tmp = merged_df.loc[merged_df['pdb_chain'].isin(chain_matches)].sort_values(by=["Resolution","Ligand MW"], ascending=True)
					#print tmp.loc[:,["pdb_chain","Resolution","Ligand ID","Ligand MW", "Exp. Method"]]
					best_match = tmp["pdb_chain"].iloc[0]
					chain_matches = ";".join(chain_matches) 
			#print best_match

			col_comp.append(comp_id)
			col_var.append(var_id)
			col_up.append(up)
			col_score.append(score)
			col_upseq.append(upseq)
			col_frac.append(float(score)/len(upseq))
			col_chains.append(chain_matches)
			col_best.append(best_match)

	chemblTarget2pdb_df = pd.DataFrame({"chembl_component_id": col_comp,
																			"chembl_variant_id": col_var,
																			"chembl_uniprot":col_up, 
																			"chembl_seq":col_upseq, 
																			"aln_score":col_score, 
																			"frac_match":col_frac, 
																			"pdb_chain_matches":col_chains, 
																			"best_match":col_best})

	chemblTarget2pdb_df.to_csv("chembl_target_uniprot_to_pdb_chain.csv",encoding="utf8")
else:
	chemblTarget2pdb_df = pd.read_csv("chembl_target_uniprot_to_pdb_chain.csv")



# In[25]:


chemblTarget2pdb_df
#
#
## In[26]:
#
#
tar_with_pdb = targets.merge(chemblTarget2pdb_df.loc[:,["chembl_component_id","frac_match","best_match"]], how="left", left_on="component_id", right_on="chembl_component_id")
tar_with_pdb.to_csv("chembl241_pchembl_single_protein_targets_uniprot_pdb.csv")
#tar_with_pdb = act_with_pdb.drop_duplicates()
#
#
## In[27]:
#
#
#

axis=activities['pchembl_value'].hist(bins=50)
axis.set_xlabel('pchembl value')
plt.show()
##pChEMBL value: https://www.ebi.ac.uk/chembl/faq#faq67
#
#
## In[28]:
#
#
#class pChEMBL_Mapper():
#	def __init__(self, lower=5.2, upper=6.0):
#		self.lower = lower
#		self.upper = upper
#	def binary_pchembl(self, x):
#		if x <= self.lower:return 0
#pmapper = pChEMBL_Mapper(lower=5.0, upper=6.0) 
#
#activities["pchembl_class"] = activities["pchembl_value"].map(pmapper.binary_pchembl)
##has_pdb = tar_with_pdb[  ~tar_with_pdb["best_match"].isin(["NO_MATCH","INSUFFICIENT_MATCH"])  ]
#activities["pchembl_class"].hist()
#
#
## In[29]:
#
#
#targets_inactive = has_pdb[has_pdb["pchembl_class"]==0]["best_match"].unique()
#targets_active = has_pdb[has_pdb["pchembl_class"]==2]["best_match"].unique()
#targets_intersect = np.intersect1d(targets_inactive, targets_active)
#
#molecules_inactive = has_pdb[has_pdb["pchembl_class"]==0]["molregno"].unique()
#molecules_active = has_pdb[has_pdb["pchembl_class"]==2]["molregno"].unique()
#molecules_intersect = np.intersect1d(molecules_inactive, molecules_active)
#
#table = pd.DataFrame([[len(targets_inactive), len(targets_active), len(targets_intersect)],
#         [len(molecules_inactive), len(molecules_active), len(molecules_intersect)]
#        ], index=["targets","molecules"], columns=["inactive set", "active set", "intersect"]
#                     )
#table
#
#
## In[30]:
#
#
#in_intersect = has_pdb[ (has_pdb["best_match"].isin(targets_intersect)) & (has_pdb["molregno"].isin(molecules_intersect))]
#
#
## In[31]:
#
#
#in_intersect["pchembl_class"].hist()
#
#
## In[32]:
#
#
#print in_intersect.shape
#print in_intersect.drop_duplicates(subset="activity_id").shape
#
#
## In[33]:
#
#
## TODO why so many redundant activities?
#
## REMOVE DUPLICATE ACTIVITY IDs - Yes or No?
#if False:
#    nr = in_intersect.drop_duplicates(subset="activity_id")
#    nr.to_csv("chembl_act_traintest_set_nr.csv", encoding="utf8)")
#    print nr["molregno"].unique().size
#    nr["pchembl_class"].hist()
#else:
#    in_intersect.to_csv("chembl_act_traintest_set.csv", encoding="utf8)")
#    
#
#
## In[40]:
#
#
#in_intersect
#
#
## In[38]:
#
#
#plt.rcParams['figure.figsize']=(16,16)
#sns.set(style="darkgrid")
##ax = sns.countplot(x="uniprot_accession", data=in_intersect, hue="protein_family_level1")
#ax = sns.factorplot(x="uniprot_accession", data=in_intersect, col="protein_family_level1",kind="count",col_wrap=2, sharex=False,sharey=False)
#
#
## In[39]:
#
#
#ax = sns.factorplot(x="uniprot_accession", data=in_intersect[in_intersect["protein_family_level1"]=="Enzyme"], col="protein_family_level2",kind="count",col_wrap=2, sharex=False,sharey=False)
#
