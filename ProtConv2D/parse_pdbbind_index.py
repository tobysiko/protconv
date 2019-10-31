# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:19:41 2018

@author: ts149092
"""
import pandas as pd
import re
import numpy as np




#print df_name.head(n=5)
#print df_name.shape




id2data={"PDB code":[], "resolution":[], "release_year":[],"measurement_type":[], "modifier":[], "binding_value":[],"binding_unit":[], "reference":[], "ligand name":[]}

fname="/nasuni/projects/ts149092/pdbbind/index/INDEX_general_PL.2016"

with open(fname) as f:
    for line in f.readlines():
        if line[0]!='#':
            id2data["PDB code"].append( line[:4] )
            resolution = line[6:10].strip()
            if resolution == "NMR": resolution = -1.0
            
            id2data["resolution"].append( float(resolution) )
            id2data["release_year"].append( int(line[12:16]) )
            
            
            
            binding_data = line[18:32].strip()
            match = re.match(r"(Kd|Ki|IC50)([<>=~]+)([0-9.]+)([a-zA-Z]+)", binding_data, re.I)
            assert match, line
            mtype, mod, val,unit=match.groups()

            id2data["measurement_type"].append( mtype )
            id2data["modifier"].append( mod )
            id2data["binding_value"].append( float(val) )
            id2data["binding_unit"].append( unit )
            id2data["reference"].append( line[35:43] )
            id2data["ligand name"].append( line[44:].strip() )

df_gen = pd.DataFrame.from_dict(id2data)
df_gen.set_index(keys=["PDB code"], drop=True, append=False, inplace=True,
                  verify_integrity=True)
#print df_gen.head(n=5)
#print df_gen.shape


id2data={"PDB code":[], "Uniprot ID":[], "protein name":[]}

fname="/nasuni/projects/ts149092/pdbbind/index/INDEX_general_PL_name.2016"

with open(fname) as f:
    for line in f.readlines():
        if line[0]!='#':
            id2data["PDB code"].append( line[:4] )
            id2data["Uniprot ID"].append( line[12:18] )
            id2data["protein name"].append( line[20:].strip() )

df_gen_name = pd.DataFrame.from_dict(id2data)
df_gen_name.set_index(keys=["PDB code"], drop=True, append=False, inplace=True,
                  verify_integrity=True)
#print df_gen_name.head(n=5)
#print df_gen_name.shape


id2data={"PDB code":[], "neg_log_meas":[]}

fname="/nasuni/projects/ts149092/pdbbind/index/INDEX_general_PL_data.2016"

with open(fname) as f:
    for line in f.readlines():
        if line[0]!='#':
            id2data["PDB code"].append( line[:4] )
            id2data["neg_log_meas"].append( float(line[19:23]) )

df_gen_data = pd.DataFrame.from_dict(id2data)
df_gen_data.set_index(keys=["PDB code"], drop=True, append=False, inplace=True,
                  verify_integrity=True)
#print df_gen_data.head(n=5)
#print df_gen_data.shape

df_general = df_gen.merge(df_gen_name, left_index=True, right_index=True, how='left')
df_general = df_general.merge(df_gen_data, left_index=True, right_index=True, how='left')
df_general["is_refined"] = False



refined_set=[]

fname="/nasuni/projects/ts149092/pdbbind/index/INDEX_refined_name.2016"

with open(fname) as f:
    for line in f.readlines():
        if line[0]!='#':
            refined_set.append( line[:4] )

df_general.loc[refined_set,"is_refined"] = True

print "GENERAL"
print df_general.head(n=5)
print df_general.shape
print df_general["is_refined"].unique()

df_general.to_csv("PDBBIND_general_PL_metadata.csv")

