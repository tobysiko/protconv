# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:30:46 2017

@author: ts149092
"""
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import subprocess as sub
import os, sys, argparse, math, random, glob, operator, json
import numpy as np
import numpy.ma as ma
import pandas as pd
import prody
import tables
from scipy.misc import imread,imsave,imresize
from ProtConv2D.MLhelpers import print_progress
#%matplotlib inline
import seaborn as sns
sns.set()

#from Bio.PDB import PDBParser, PDBExceptions, PDBIO, Selection, Residue


class PDBToImageConverter():
    def __init__(self, data_path, pdb_path, png_path, npy_path, id_list, 
                 img_dim=None,   overwrite_files=False, selection="protein and heavy", 
                 pdb_suffix=".pqr", channels=["dist","anm","electro"], 
                 output_format="png", save_channels=True, randomize_list=False, 
                 distmat_cutoff=51.2, invert_img=False, maxnatoms=20000,
                 is_pdbbind=True, sample_modes=False):
        self.data_path = data_path
        if not os.path.exists(data_path): os.mkdir(data_path)
        self.pdb_path = pdb_path
        if not os.path.exists(pdb_path): os.mkdir(pdb_path)
        self.png_path = png_path
        if not os.path.exists(png_path): os.mkdir(png_path)
        self.npy_path = npy_path
        if not os.path.exists(npy_path): os.mkdir(npy_path)
        self.id_list = id_list
        self.img_dim = img_dim
        self.overwrite_files = overwrite_files
        self.selection = selection
        self.pdb_suffix = pdb_suffix
        self.channels = channels
        self.output_format = output_format
        self.save_channels = save_channels
        self.distmat_cutoff = distmat_cutoff
        self.invert_img = invert_img
        self.maxnatoms = maxnatoms
        self.is_pdbbind = is_pdbbind
        self.sample_modes = sample_modes
        if randomize_list:
            tmp=len(id_list)
            random.shuffle(id_list)
            assert len(id_list)==tmp and len(id_list) > 0, str(len(id_list))
            
    
    def get_anm_crosscorr(self, prot, name="prot"):
        try:
            anm, sel = prody.calcANM(prot, selstr=self.selection)
            if self.sample_modes:
                ens = prody.sampleModes(anm,sel,n_confs=10, rmsd=0.8)
                prody.writePDB(name, ens)
            #print anm,sel
            anm.calcModes()
            cc = prody.calcCrossCorr(anm)
            #prody.view3D(prot, flucts=anm)
            return cc
        except:
            print( "ANM failed!")
            return np.array([])
    
    def save_matrix_npy(self, matrix, label):
        savename = os.path.join(self.npy_path, label+".npy")
        #https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html
        if not os.path.exists(savename) or self.overwrite_files:
            np.save(savename, matrix, allow_pickle=True)
        return savename
    
    def load_matrix_np(self, filename):
        return np.load(filename, allow_pickle=True)    
        
    def norm_mat_img(self, matrix):
        if self.invert_img:
            return 256+(-255*(matrix-np.amin(matrix))/np.amax(matrix))
        else:
            return 255*(matrix-np.amin(matrix))/np.amax(matrix)
    
    def save_matrix_1c_png(self, matrix, label):
        savename = os.path.join(self.png_path,label+".png")
        if not os.path.exists(savename) or self.overwrite_files:
            #mpl.use('Agg')
            matrix = self.norm_mat_img(matrix)
            if self.img_dim!=None:
                #print matrix
                
                matrix = imresize(matrix, (self.img_dim, self.img_dim),interp='nearest',mode='L')
            imsave(savename, matrix)
        return savename
    
    def load_matrix_1c_png(self, filename):
        return imread(filename, mode="L", flatten=True)
    
    def load_matrix_3c_png(self, filename):
        return imread(filename, mode="RGB", flatten=False)
    
    def save_matrix_1c_png_mpl(self, matrix, label): 
        savename = os.path.join(self.png_path,label+".png")
        if not os.path.exists(savename) or self.overwrite_files:
            #mpl.use('Agg')
            matrix = self.norm_mat_img(matrix)
            plt.Figure(figsize=(self.img_dim,self.img_dim),dpi=1)
            ax = plt.axes([0,0,1,1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False) 
            
            plt.imshow(matrix,aspect='equal', origin='lower', interpolation='nearest')
            plt.axis('off')
            plt.savefig(savename, bbox_inches='tight', pad_inches=0,dpi='figure')
            plt.clf()
        return savename
    
    def make_rgb_tensor(self, r, g, b):
        r = self.norm_mat_img(r)
        g = self.norm_mat_img(g)
        b = self.norm_mat_img(b)
        
        rgb = np.zeros((r.shape[0], r.shape[1], 3), "uint8") 
        rgb [:,:,0] = r
        rgb [:,:,1] = g
        rgb [:,:,2] = b
        return rgb

    def save_matrix_3c_png(self, rgb, label):
        rgbname = os.path.join(self.png_path, label+".png")
        
        imsave(rgbname, rgb)
        return rgbname
    
    def save_matrix_4c_png(self, label, r, g, b, a):
        rgbaname = os.path.join(self.png_path, label+".png")
        r = self.norm_mat_img(r)
        g = self.norm_mat_img(g)
        b = self.norm_mat_img(b)
        a = self.norm_mat_img(a)
    
        rgba = np.zeros((r.shape[0], r.shape[1], 4), "uint8") 
        rgba [:,:,0] = r
        rgba [:,:,1] = g
        rgba [:,:,2] = b
        rgba [:,:,3] = a
        imsave(rgbaname, rgba)
        return rgbaname
    
    def pdb2electro(self, prot, distmat, mincut=-0.1, maxcut=0.1):# requires data from PQR file!
        n = len(prot)
        nbmat = np.zeros((n,n))
        charges = prot.getCharges()
        radii = prot.getRadii()
        #pafu = 0
        for i in range(n):
            q_i=charges[i]
            #r_i=radii[i]
            for j in range(n):
                if i<j:
                    q_j=charges[j]
                    #r_j=radii[j]
                    r_ij=distmat[i,j]
                    
                    if r_ij > 0.0:# and r_ij<=25.6:
                        
                        #https://en.wikipedia.org/wiki/AMBER
                        
                        sig = 12.0 # distance where potential is zero
                        r0_ij = 1.122*sig # distance where potential at minimum
                        r6 = math.pow(r0_ij, 6)
                        V_ij = 1.0#-120.0
                        e_r = 1.0*math.pi*4
                        f=1.0#138.935458
                        A_ij = 2*r6*V_ij
                        B_ij = (A_ij/2.0)*r6
                        E_LJ = (-A_ij)/math.pow(r_ij,6) + (B_ij)/math.pow(r_ij,12)
                        E_coul = f*(q_i*q_j) / (e_r*r_ij)
                        E_nb = E_LJ + E_coul
                        E_nb =  min(maxcut,max(mincut, E_nb ))
                        #E_nb = np.exp(-(E_LJ + E_coul)/1.0  )
                        
                        #E_nb = -(E_LJ + E_coul)
                        #if r_ij < 6.0:
                        #    E_nb = -E_nb
                    else:
                        E_nb = 0.0
                        #E_nb = 1.0
                    nbmat[i,j] = E_nb
                    #pafu += E_nb
                    #if i!=j: 
                    nbmat[j,i] = E_nb
                    #pafu += E_nb
                    #nbmat = nbmat+np.min(nbmat)
                    #nbmat = nbmat-np.min(nbmat)
                    #nbmat = np.amin(10,nbmat)
                    #nbmat = nbmat/np.max(nbmat)
                    #print np.min(nbmat), np.max(nbmat), np.median(nbmat)
        return nbmat#/pafu
    
    def pdb2mat(self, pdb, channels=["dist","anm","electro"]):
        
        if self.pdb_suffix==".pqr":
            prot = prody.parsePQR(pdb).select(self.selection)
        else:
            prot = prody.parsePDB(pdb).select(self.selection)
        
        matrices = {}
        distmat = prody.buildDistMatrix(prot)
        for ch in channels:
            if ch=="dist":
                matrices[ch] = distmat
            if ch=="anm":
                ccmap = self.get_anm_crosscorr(prot,name=pdb.strip(self.pdb_suffix)+"_modes.pdb")
                matrices[ch] = ccmap
            if ch=="electro":
                nbmat = self.pdb2electro
                matrices[ch] = nbmat
            if ch=="anmElectro":
                
                ccmap = self.get_anm_crosscorr(prot,name=pdb.strip(self.pdb_suffix)+"_modes.pdb")
                nbmat = self.pdb2electro
                
                matrices[ch] = matrices[ch]=np.array( [[ccmap[i,j] if i>j else nbmat[i,j] for j in range(ccmap.shape[0])] for i in range(ccmap.shape[0])] )
            if ch=="dfk1":
                matrices[ch] = np.power(distmat, -2*1)
            if ch=="dfk2":
                matrices[ch] = np.power(distmat, -2*2)
            if ch=="dfk3":
                matrices[ch] = np.power(distmat, -2*3)
            if ch=="dfk4":
                matrices[ch] = np.power(distmat, -2*4)
        return matrices
    
    # Main function. Does the actual conversion from PDB/PQR file to image/array file.
   
    def convert_pdbs_to_arrays(self, verbosity=0, show_pairplot=False, show_distplots=False, show_matrices=False, show_protein=False):
        n = len(self.id_list)
        
        print ("Processing channels:", self.channels)
        if self.pdb_suffix==".pqr": parser = prody.parsePQR
        else: parser = prody.parsePDB
        
        if show_protein:
            import py3Dmol
        
        print_progress(0, n, prefix='converting structures:', suffix='', decimals=2, bar_length=20, prog_symbol='O')
        
        for di in range(n):
            dom = self.id_list[di]
            if show_protein:
                q='pdb:%s'%dom[:4]
                
                p3d = py3Dmol.view(query=q)
                p3d.setStyle({'cartoon': {'color':'spectrum'}})
                p3d.show()
            
            if self.is_pdbbind:
                p = os.path.join(self.pdb_path, dom, "%s_protein%s"%(dom,self.pdb_suffix))
                pp = os.path.join(self.pdb_path,dom, "%s_pocket%s"%(dom, ".pdb")) ## currently not using PQR for pockets!!!!
            else:
                p = os.path.join(self.pdb_path, "%s%s"%(dom,self.pdb_suffix))
                pp=None
            
            if not os.path.exists(p): 
                if verbosity>0:print( "Error: could not find ",p)
                continue
            
            
        
        
            matrices = {}
            distmat = np.array([])
            #dm_I = np.array([])
            #dm_mask = np.array([])
            prot = None
            nat=0
            rgblabel = dom+"_rgb" 
            if self.output_format=="npy":
                rgbfile = os.path.join(self.npy_path, rgblabel+".npy")
            else:
                rgbfile = os.path.join(self.png_path, rgblabel+".png")
            
            
            if not self.overwrite_files:
                is_missing = {}
                for ch in self.channels:
                    if self.output_format=="png":
                        tmppath = os.path.join(self.png_path, dom+"_%s.png"%ch)
                    elif self.output_format=="npy":
                        tmppath = os.path.join(self.npy_path, dom+"_%s.npy"%ch)
                    else:
                        tmppath = None
                    
                    if not os.path.exists(tmppath):
                        is_missing[ch] = True
                    else:
                        is_missing[ch] = False
                # if all files are there, no need to continue
                if not any([is_missing[k] for k in is_missing.keys()]) and os.path.exists(rgbfile):
                    continue
            
            for ch in self.channels:
                if self.overwrite_files or is_missing[ch]:
                    if prot==None: 
                        
                        try:
                            prot = parser(p).select(self.selection, quiet=True)
                            pocket = None
                            if self.is_pdbbind:
                                pocket = parser(pp).select(self.selection, quiet=False)
                                
                        except AttributeError as ae:
                            print (dom, ch, ae)
                            continue
                        #for p in prot.get
                        nat = prot.numAtoms()
                        if nat > self.maxnatoms:
                            prot=None
                            print( "skipping %s: exceeds self.maxnatoms=%i. "%(dom,self.maxnatoms),nat)
                            continue
                        if show_protein:
                            prody.showProtein(prot)
                        
                        
                    if distmat.shape != (nat,nat): 
                        if self.is_pdbbind:
                            distmat, distmat_pocket = buildDistMatrixAnnot(prot, atoms2=None, annot_group=pocket, unitcell=None, format='mat')
                        else:
                            distmat = prody.buildDistMatrix(prot)
                            distmat_pocket = None
                        
                        
#                        dm_I=np.eye(distmat.shape[0])
#                        dm_mask = ma.masked_array(distmat, mask=dm_I)
                        assert distmat.shape == (nat,nat)

                    print_progress(di+1, n, prefix='converting structures:', suffix=' - %s - %i atoms - calculating: %s'%(dom,nat,ch.ljust(12)), decimals=2, bar_length=20, prog_symbol='O')

                    anm = None
                    if ch=="dist":
                        mat = distmat.copy()
                        if self.distmat_cutoff!=-1:
                            mat[mat>self.distmat_cutoff] = self.distmat_cutoff
                            matrices[ch] = mat / self.distmat_cutoff
                        else:
                            matrices[ch] = mat
                    elif ch=="pocket":
                        mat = distmat_pocket.copy()
                        if self.distmat_cutoff!=-1:
                            mat[mat>self.distmat_cutoff] = self.distmat_cutoff
                            matrices[ch] = mat / self.distmat_cutoff
                        else:
                            matrices[ch] = mat
                    elif ch=="anm":
                        anmcc = self.get_anm_crosscorr(prot,name=p.strip(self.pdb_suffix)+"_modes.pdb")
                        if np.ndim(anmcc)>2:
                            print ("Error: ANM failed.",ch,p)
                            continue
                        matrices[ch] = (anmcc+1.0) / 1.0
                        #prody.view3D(prot)
                    elif ch=="electro":
                        mincut =-0.5
                        maxcut = 0.05
                        mat = self.pdb2electro(prot, distmat, mincut=mincut, maxcut=maxcut)#*-1.0
                        #plt.scatter(mat.flatten(),distmat.flatten());plt.show()
                        #mat[(mat>250)] = 250
                        #mat[mat>-150] = -150
                        #mat[(mat<=0.0)&(mat>-50.0)] *=-1.0
                        #mat[(mat<-150)] += 150
                        #mat[mat<-5] = -5
                        #mat[mat>5] = 5
                        #matrices[ch] = np.log((mat+.1)/.1)#mat#(mat+5) / 5
                        matrices[ch] = (mat-mincut)/maxcut
#                    elif ch=="dfk1":
#                        matrices[ch] = np.power(dm_mask, np.full_like(distmat, -2*1)).filled(0)
#                    elif ch=="dfk2":
#                        matrices[ch] = np.power(dm_mask, np.full_like(distmat, -2*2)).filled(0)
#                    elif ch=="dfk3":
#                        matrices[ch] = np.power(dm_mask, np.full_like(distmat, -2*3)).filled(0)
#                    elif ch=="dfk4":
#                        matrices[ch] = np.power(dm_mask, np.full_like(distmat, -2*4)).filled(0)
                    elif ch=="mechstiff":
                        anm = prody.ANM('anm '+dom)
                        anm.buildHessian(prot, cutoff=13.0)
                        anm.calcModes(n_modes=20)
                        stiffness = prody.dynamics.mechstiff.calcMechStiff(anm ,prot)
                        matrices[ch] = stiffness#(stiffness+1.0) / 1.0
                    elif ch=="anmElectro":
                        anmcc = self.get_anm_crosscorr(prot,name=p.strip(self.pdb_suffix)+"_modes.pdb")
                        if np.ndim(anmcc)>2:
                            print ("Error: ANM failed.",ch,p)
                            continue
                        anmcc = (anmcc+1.0) / 1.0
                        
                        anm = self.norm_mat_img( anmcc )
                        
                        mincut =-0.5
                        maxcut = 0.05
                        nb = self.norm_mat_img( (self.pdb2electro(prot, distmat, mincut=mincut, maxcut=maxcut)-mincut) / maxcut )
                        
                        matrices[ch]=np.array( [[anm[i,j] if i>j else nb[i,j] for j in range(nat)] for i in range(nat)] )
                    else:
                        print( "Unknown channel:",ch)
                        sys.exit(1)
                    
                    if show_distplots:
                        
                        histo = sns.distplot(matrices[ch].flatten(),bins=256,kde=False, color='r')
                        histo.set_yscale('log')
                        plt.xlabel("normalized values");plt.ylabel("log counts");plt.title("matrix histogram: "+ch);plt.show()
                    
                    if nat>0 and matrices[ch].shape==(nat,nat) and self.save_channels:
                        
                        if self.output_format=="npy":
                            savename = self.save_matrix_npy(matrices[ch], dom+"_"+ch)
                        else:
                            savename = self.save_matrix_1c_png(matrices[ch], dom+"_"+ch)
                        assert os.path.exists(savename)
                else:
                    
                    if verbosity > 0: print( "loading",ch)
                    if self.output_format=="npy":
                        matrices[ch] = self.load_matrix_np(os.path.join(self.npy_path, dom+"_%s.npy"%ch))
                        
                    else:
                        matrices[ch] = self.load_matrix_1c_png(os.path.join(self.png_path, dom+"_%s.png"%ch))
                    print_progress(di+1, n, prefix='converting structures:', suffix=' - %s - %i atoms - from file: %s'%(dom,matrices[ch].shape[0],ch.ljust(12)), decimals=2, bar_length=20, prog_symbol='O')
                
                if show_matrices: 
                    cax=plt.imshow(self.norm_mat_img(matrices[ch]), interpolation="none");plt.gca().grid(False);plt.colorbar(cax);plt.title("atomic pairwise matrix (%s); %s"%(ch, self.selection));plt.show()
            if len(matrices)==3 and all([matrices[ch].shape[0] > 0 for ch in self.channels]):
                if show_pairplot:
                    #tmpdf = pd.DataFrame(data={"dist":matrices["dist"].flatten(),
                    #                        "electro":matrices["electro"].flatten(),
                    #                        "anm":matrices["anm"].flatten()
                    #                        })
                    tmpdf = pd.DataFrame(data={ch:matrices[ch].flatten() for ch in self.channels
                                            })
                    pairplot_n = len(tmpdf.index)/10
                    sns.pairplot(tmpdf.ix[random.sample(tmpdf.index, pairplot_n)]);plt.title("%i samples from 2D matrix"%(pairplot_n));plt.show()
                print_progress(di+1, n, prefix='converting structures:', suffix=' - %s - %i atoms - writing: %s'%(dom,matrices[ch].shape[0],"RGB".ljust(12)), decimals=2, bar_length=20, prog_symbol='O')
                
                rgb = self.make_rgb_tensor( matrices[self.channels[0]], 
                                            matrices[self.channels[1]], 
                                            matrices[self.channels[2]]
                                            )
                if show_matrices: 
                    plt.imshow(rgb,  interpolation="none"); plt.gca().grid(False);plt.title("atomic pairwise matrix (RGB); %s"%(self.selection));plt.show()
                if self.overwrite_files or not os.path.exists(rgbfile):
                    if self.output_format=="npy":
                        savename = self.save_matrix_npy(rgb, rgblabel)
                    else:
                        savename = self.save_matrix_3c_png(rgb, rgblabel)
                    assert os.path.exists(savename)

def overlapping_tiles_from_images(img_folder, tiles_folder, tile_size=512, n_channels=3, suffix="_rgb.png", show_plots=False, write_individual_tiles=False):
    glob_string = os.path.join(img_folder, "*%s"%(suffix))
    print(glob_string)
    files = glob.glob( glob_string )
    print("%i files found"%len(files))

    #h5_img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved

    for f in files[:1]:
        identifier = os.path.basename(f).split(".")[0]
        
        if n_channels==3:
            img = imread(f, flatten=False, mode="RGB")
        else:
            img = imread(f, flatten=True)
        
        if n_channels == 1:
            x = np.array(img)[..., np.newaxis]
        else:
            x = np.array(img)
        
        print(identifier, x.shape)
        
        dim = x.shape[0]
        if show_plots: plt.imshow(x, interpolation="none"); plt.gca().grid(False);plt.title("full image");plt.show()

        n_splits = math.ceil(dim/float(tile_size))
        start = math.floor(dim/n_splits)

        tiles_per_image = []
        hdf5_path = os.path.join(tiles_folder,"%s_tile%i.h5"%(identifier, tile_size))
        
        coords=[]
        counter = 0
        for i in range(n_splits):
            for j in range(n_splits):
                #print(i,j, n_splits, start)
                xij = np.zeros((tile_size, tile_size,n_channels), dtype=np.int)

                tile = x[i*start:i*start+tile_size, j*start:j*start+tile_size,:]
                
                xij[:tile.shape[0],:tile.shape[1],:] = tile

                #print(xij.shape)
                
                if show_plots: plt.imshow(xij, interpolation="none"); plt.gca().grid(False);plt.title("%i,%i"%(i,j));plt.show()
                coords.append((i*start, j*start))
                tiles_per_image.append(xij)
                if write_individual_tiles: imsave(os.path.join(tiles_folder,"%s_tile%i_%i_%i_%i%s"%(identifier, tile_size, counter, i*start, j*start, suffix)), xij)
                counter += 1
        hdf5_file = tables.open_file(hdf5_path, mode='w')
        a = np.array(tiles_per_image, dtype=np.uint8)
        print(a.shape)
        hdf5_file.create_array(hdf5_file.root, 'tiles', a )
        hdf5_file.create_array(hdf5_file.root, 'xy', coords)
        hdf5_file.close()

                













            




# adapted from ProDy
def buildDistMatrixAnnot(atoms1, atoms2=None, annot_group=None, unitcell=None, format='mat'):
    """Returns distance matrix.  When *atoms2* is given, a distance matrix
    with shape ``(len(atoms1), len(atoms2))`` is built.  When *atoms2* is
    **None**, a symmetric matrix with shape ``(len(atoms1), len(atoms1))``
    is built.  If *unitcell* array is provided, periodic boundary conditions
    will be taken into account.

    :arg atoms1: atom or coordinate data
    :type atoms1: :class:`.Atomic`, :class:`numpy.ndarray`

    :arg atoms2: atom or coordinate data
    :type atoms2: :class:`.Atomic`, :class:`numpy.ndarray`

    :arg unitcell: orthorhombic unitcell dimension array with shape ``(3,)``
    :type unitcell: :class:`numpy.ndarray`

    :arg format: format of the resulting array, one of ``'mat'`` (matrix,
        default), ``'rcd'`` (arrays of row indices, column indices, and
        distances), or ``'arr'`` (only array of distances)
    :type format: bool"""
    
    prot_atoms = atoms1
    #print atoms1.getResnums()
    #print atoms1.getSequence()
    #print atoms1.getChids()
    

    #print annot_group.getResnums()
    #print annot_group.getSequence()
    
    annot_resnums = annot_group.getResnums()
    annot_chains = annot_group.getChids()
    #print annot_chains
    
    if ' ' in annot_chains: # WHAT DOES THIS MEAN?
        unique_chains = pd.Series(prot_atoms.getChids()).unique()
        annot_chains = [unique_chains[0] for _ in annot_chains] 
#         annot_chains_new = []
#         annot_resnums_new = []
#         for c in unique_chains:
#             annot_chains_new.extend( [c for _ in annot_resnums ] )
#             annot_resnums_new.extend(annot_resnums)
#         print annot_chains_new
#         print annot_resnums_new
#         annot_chains = annot_chains_new
#         annot_resnums = annot_resnums_new
        #print annot_chains
                
    
    annot_rc_pairs = zip(annot_resnums, annot_chains )
    atoms1_is_pocket = np.zeros(len(atoms1))
    for i, xyz_i in enumerate(atoms1):
            prot_res_num_i = prot_atoms[i].getResnum()
            prot_res_chain_i = prot_atoms[i].getChid()
            if (prot_res_num_i, prot_res_chain_i) in annot_rc_pairs: 
                atoms1_is_pocket[i]=1
    
    
    
    
    if not isinstance(atoms1, np.ndarray):
        try:
            atoms1 = atoms1._getCoords()
        except AttributeError:
            raise TypeError('atoms1 must be Atomic instance or an array')
    if atoms2 is None:
        symmetric = True
        atoms2 = atoms1
        
        
        
        
    else:
        symmetric = False
        if not isinstance(atoms2, np.ndarray):
            try:
                atoms2 = atoms2._getCoords()
            except AttributeError:
                raise TypeError('atoms2 must be Atomic instance or an array')
    if atoms1.shape[-1] != 3 or atoms2.shape[-1] != 3:
        raise ValueError('one and two must have shape ([M,]N,3)')

    if unitcell is not None:
        if not isinstance(unitcell, np.ndarray):
            raise TypeError('unitcell must be an array')
        elif unitcell.shape != (3,):
            raise ValueError('unitcell.shape must be (3,)')

    dist = np.zeros((len(atoms1), len(atoms2)))
    dist_annot = np.zeros((len(atoms1), len(atoms2)))
    if symmetric:
        if format not in prody.measure.measure.DISTMAT_FORMATS:
            raise ValueError('format must be one of mat, rcd, or arr')
        if format == 'mat':
            for i, xyz_i in enumerate(atoms1):
                
                
                for j, xyz_j in enumerate(atoms2):
                    
                    if i>j:
                        
                        dist[i,j] = dist[j,i] =  prody.measure.measure.getDistance(xyz_i, xyz_j, unitcell)
                        if  atoms1_is_pocket[i] and atoms1_is_pocket[j]:

                            dist_annot[i, j] = dist_annot[j, i] = dist[i, j]
                            
        else:
            dist = np.concatenate([prody.measure.measure.getDistance(xyz, atoms2[i+1:], unitcell)
                                for i, xyz in enumerate(atoms1)])
            if format == 'rcd':
                n_atoms = len(atoms1)
                rc = np.array([(i, j) for i in range(n_atoms)
                            for j in range(i + 1, n_atoms)])
                row, col = rc.T
                dist = (row, col, dist)

    else:
        for i, xyz in enumerate(atoms1):
            dist[i] = prody.measure.measure.getDistance(xyz, atoms2, unitcell)
    return dist, dist_annot


#  Obtain domain list from file. Domain IDs are the first white-space separated entry in a line. Comments (#...) are ignored.            
def getDomList(loc):
    with open(loc, 'r') as domlistfile:
        lines = domlistfile.readlines()#.split()
    idlist = []
    for l in lines:
        line = l#.decode("utf-8-sig")
        linesplit=line.split()
        if line[0] != '#' and len(linesplit)>0:
            idlist.append(linesplit[0])
    return idlist

# Run external executable of "pdb2pqr" on PDB files from a domain list within a specified folder and converts them to PQR files in a different folder.
def pdb2pqr(idlist, pdbloc,pqrloc,exe="C:/Users/ts149092/Documents/pdb2pqr-windows-bin64-2.1.0/pdb2pqr.exe",show_output=False, show_error=False, ff="amber", quiet=False, is_pdbbind=False):
    if not quiet: print_progress(0,len(idlist), prefix="Convert PDB to PQR")
    for di in range(len(idlist)):
        dom = idlist[di]
        if is_pdbbind:
            dompdb = os.path.join(pdbloc,dom,"%s_protein.pdb"%dom)
            dompqr = os.path.join(pdbloc,dom,"%s_protein.pqr"%dom)
        else:
            dompdb = os.path.join(pdbloc,"%s.pdb"%dom)
            dompqr = os.path.join(pqrloc,"%s.pqr"%dom)
        if not quiet: print( dompdb, dompqr, os.path.exists(dompdb) and not os.path.exists(dompqr) )
        if os.path.exists(dompdb) and not os.path.exists(dompqr):
            p = sub.Popen("%s --ff=%s %s %s"%(exe, ff, dompdb, dompqr), shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
            err=p.stderr.read()
            out=p.stdout.read()
            if show_output: 
                print( "<OUTPUT %s>"%dompdb, out)
            if show_error and err.strip()!="":                 
                print( "<ERROR %s>"%dompdb, err )
            
            p.wait()
        else:
            if not quiet: print (" - %s exists? %i, %s exists? %i"%(dompdb, int(os.path.exists(dompdb)), dompqr,  int(os.path.exists(dompqr))) )
        try:
            if not quiet: print_progress(di,len(idlist),prefix="Convert PDB to PQR",suffix=dom)
        except UnicodeDecodeError as ude:
            print( di)
            print( dom )
            print( ude)

# Parses a line from a PDB/PQR file and if it is an ATOM or HETATM entry, returns it as a dictionary or tuple.
def parseATOMline(line, returnDict=True, is_pqr=False): 
    d = {}
    d["ltype"] = line[:6]
    try:
        d["anum"] = int(line[6:11])
        d["aname"] = line[12:16].strip()
        d["aname_exact"] = line[12:16]
        d["altloc"] = line[16]
        d["rname"] = line[17:20].strip()
        d["chain"] = line[21]
        d["rnum"] = int(line[22:26])
        d["insert"] = line[26]
        if d["ltype"] in ["ATOM  ", "HETATM"]:
            d["x"] = float(line[30:38])
            d["y"] = float(line[38:46])
            d["z"] = float(line[46:54])
            
            if is_pqr:
                tmp = line[54:].strip().split()
                assert len(tmp)==2
                d["occupancy"] = float(tmp[0])
                d["bfactor"] = float(tmp[1])
            else:
                d["occupancy"] = float(line[54:60])
                d["bfactor"] = float(line[60:66])
                if len(line)>=66:
                    d["segment"] = line[72:76].strip()
                    d["element"] = line[76:78].strip()
                    d["charge"] = line[78:80].strip()
    except ValueError as ve:
        print( ve)
        print( line)
        sys.exit(1)
    if returnDict:
        return d
    else:
        if is_pqr:
            return (d["ltype"], d["anum"], d["aname"], d["altloc"], d["rname"], d["chain"], d["rnum"], d["insert"], d["x"], d["y"], d["z"], d["occupancy"], d["bfactor"])
        else:
            return (d["ltype"], d["anum"], d["aname"], d["altloc"], d["rname"], d["chain"], d["rnum"], d["insert"], d["x"], d["y"], d["z"], d["occupancy"], d["bfactor"], d["segment"], d["element"], d["charge"])


# Converts any PDB/PQR file to a pandas data frame. Note that for PQR, "occupancy" is the partial atomic charge and "bfactor" is the atom radius, and these are the final two entries in each ATOM row.
def pdb_to_dataframe(pdbfilename, col_labels=["ltype", "anum", "aname", "altloc", "rname", "chain", "rnum", "insert", "x", "y", "z", "occupancy", "bfactor", "segment", "element", "charge"], is_pqr=False):

    atomlines=[]
    with open(pdbfilename) as pdbfile:
        lines = pdbfile.readlines()
        
        for l in lines:
            if l[:6]=="ATOM  ":
                atomlines.append( list(parseATOMline(l, returnDict=False, is_pqr=is_pqr)) )
    atomlines = np.array(atomlines).T
    if is_pqr:
        col_labels = col_labels[:13]
    data = {col_labels[li]:atomlines[li] for li in range(len(col_labels))}
    df = pd.DataFrame(data=data,columns=col_labels )
    
    return df
         
# Go through all PDB/PQR files in the folder and check for consistent atom lists per residue.
# Outputs a dictionary ("atomchecker") of resiude names in 3-letter code mapped to dictoinary mapping a specific atom name tuple ( order as found in file) to number of observations
def check_atom_order_pdbs(path, pdb_suffix, idlength=7, is_pqr=False):
    fp = os.path.join(path, "%s%s"%(idlength*"?", pdb_suffix))
    AA_3_to_1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    AAs = AA_3_to_1.keys()
    files = glob.glob(fp)
    print( fp, len(files) )
    atomchecker = {aa:{} for aa in AAs}# list of atom-order tuple - frequency pairs
    print_progress(0,len(files))
    for pi in range(len(files[:10])):
        print_progress(pi,len(files))
        p = files[pi]
        #print pi, p
        #lines=[]
        df = pdb_to_dataframe(p, is_pqr=is_pqr)
        #print df
        rnums = df["rnum"].unique()
        for r in rnums:
            inserts = df[df["rnum"]==r]["insert"].unique()
            # TODO: 
            rname = df[ (df["rnum"]==r) & (df["insert"]==inserts[0]) ]["rname"].unique()
            altlocs = df[df["rnum"]==r]["altloc"].unique()
            
            if len(rname)!=1:
                print( "%s >> WARNING: multiple rnames per rnum %s! %s (altlocs: %s; inserts: %s)"%(p, r, str(rname),str(altlocs ), str(inserts) ) )
            rname = rname[0]
            aorder= str(",".join(df[df["rnum"]==r]["aname"]))
            #print r, rname
            if rname in atomchecker:
                if aorder in atomchecker[rname]:
                    atomchecker[rname][aorder] += 1
                else:
                    atomchecker[rname][aorder] = 1
            else:
                print( "Found non-standard AA:",rname )
                atomchecker[rname] = {}
                atomchecker[rname][aorder] = 1
    
    return atomchecker
                
                
                
            
        
                    
        
    
    
if __name__=="__main__":
    	# PARSE COMMAND LINE ARGUMENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--basepath", default="", help="")
    ap.add_argument("--idlistfile", default="PDBBIND_general_PL_metadata.csv", help="")
    ap.add_argument("--pdbpath", default="refined_set",help="")
    ap.add_argument("--pqrpath", default="pqr",help="")
    ap.add_argument("--pngpath", default="png2",help="")
    ap.add_argument("--h5name", default="h5out.h5",help="")
    ap.add_argument("--imgdim", type=int, default=None, help="")
    ap.add_argument("--selection", default="protein and heavy", help="a valid ProDy selection")
    ap.add_argument("--overwrite", action="store_true",default=False, help="")
    ap.add_argument("--calcpqr", action="store_true",default=False, help="")
    ap.add_argument("--informat",choices=["pdb","pqr"],default="pdb",help="pdb or pqr")
    ap.add_argument("--outformat",choices=["png","npy"],default="png",help="png or npy - beware that npy binary files are huge!")
    ap.add_argument("--pdb2pqr", default="C:/Users/ts149092/Documents/pdb2pqr-windows-bin64-2.1.0/pdb2pqr.exe",help="path to the pdb2pqr executable")
    ap.add_argument("--savechannels", type=bool, default=True, help="If three channels are provided, an rgb output will be written. Do you want to save the individual channels as well?")
    ap.add_argument("--randomize", action="store_true", help="randomize the order of inputs")
    ap.add_argument("--distmat_cutoff", type=float, default=51.2, help="")
    ap.add_argument("--show_plots", action="store_true", help="")
    ap.add_argument("--invert_img", action="store_true", help="")
    ap.add_argument("--check_order",action="store_true",help="")
    ap.add_argument("--maxnatoms", type=int, default=20000, help="")
    ap.add_argument("--pdbbind", action="store_true",help="")
    ap.add_argument("--channels", default="dist,anmElectro,pocket",help="")
    ap.add_argument("--verbosity", type=int, default=0,help="")
    
    args = ap.parse_args()
    print( "Settings:",args)
    
    # obtain list of all domains to include in the data set from file
    if args.pdbbind:
        meta_df = pd.read_csv(os.path.join(args.basepath, args.idlistfile))
        idlist = list(meta_df[meta_df["is_refined"]==True ] ["PDB code"].values)
    else:
        idlist = getDomList(os.path.join(args.basepath, args.idlistfile))
    print( len(idlist),"PDB files expected")
    
    # optionally, if no PQR files available, convert a folder of PDB files to a folder of PQR files. PQR needed when partial charges required.
    if args.calcpqr:
        print( "converting PDB files at %s to PQR files at %s"%(args.pdbpath, args.pqrpath) )
        pdb2pqr(idlist, os.path.join(args.basepath, args.pdbpath), os.path.join(args.basepath, args.pqrpath), exe=args.pdb2pqr, is_pdbbind=args.pdbbind, show_output=True, show_error=True, quiet=False)
        
    
    if args.informat=="pdb":
        suffix = ".pdb"
        pdbpath = args.pdbpath
    elif args.informat=="pqr":
        if args.pdbbind:
            pdbpath = args.pdbpath
        else:
            pdbpath = args.pqrpath
        suffix = ".pqr"
    else:
        print( "unknown input format" )
        sys.exit(1)
        
    # control ProDy's output
    #http://prody.csb.pitt.edu/manual/reference/prody.html#module-prody
    prody.confProDy(verbosity="critical") #(‘critical’, ‘debug’, ‘error’, ‘info’, ‘none’, or ‘warning’)
    
    # initialize object with all arguments
    converter = PDBToImageConverter(
                    args.basepath, 
                    os.path.join(args.basepath, pdbpath) , 
                    os.path.join(args.basepath, args.pngpath), 
                    os.path.join(args.basepath, args.h5name), 
                    idlist,
                    img_dim=args.imgdim, 
                    overwrite_files=args.overwrite,
                    selection=args.selection,
                    pdb_suffix=suffix,
                    
                    channels=args.channels.split(','),
                    #channels=["anmElectro"],
                    output_format=args.outformat,
                    save_channels=args.savechannels,
                    randomize_list=True,
                    distmat_cutoff = args.distmat_cutoff,
                    invert_img = args.invert_img,
                    maxnatoms = args.maxnatoms,
                    is_pdbbind = args.pdbbind
                 )
    
    # Go through all PDB/PQR files in the folder and check for consistent atom lists per residue.
    # Outputs a dictionary ("atomchecker") of resiude names in 3-letter code mapped to dictoinary mapping a specific atom name tuple ( order as found in file) to number of observations
    if args.check_order:
        atomchecker= check_atom_order_pdbs(os.path.join(args.basepath, pdbpath),suffix, is_pqr=True if args.informat=="pqr" else False)
        
        
        
        sorted_achecker = {}
        # print atomchecker to screen
        for aa in sorted(atomchecker.keys()):
            print( aa )
            aos = sorted(atomchecker[aa].items(), key=operator.itemgetter(1), reverse=True )
            sorted_achecker[aa] = aos
            for aord in aos:
                print( "\t%s:\t %s"%(str(aord[1]) , str(aord[0])) )
        # write atomchecker dictionary to json file
        with open('atomchecker.json', 'w') as fp:
            json.dump(sorted_achecker, fp, indent=4, sort_keys=True)
    else:
        # convert PDB/PQR files to image / numpy array
        converter.convert_pdbs_to_arrays(show_pairplot=args.show_plots, show_distplots=args.show_plots, show_matrices=args.show_plots, show_protein=args.show_plots, verbosity=args.verbosity)
    
    # TIP:
    # Split text file into n equal chunks in bash to run in parallel:
    # split -n 5 textfile.txt chunktxt
