import argparse
import getpass
import glob
import json
import math
import operator
import os
import random
import subprocess as sub
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import matplotlib as mpl

# mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import seaborn as sns
import tables
from scipy.misc import imread, imresize, imsave
from sklearn.metrics.pairwise import pairwise_distances

import pathlib
local_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(os.path.split(local_path)[0])
print(sys.path)
import utils.parallel_distmat as par
import prody
from utils.utils import print_progress

userid = getpass.getuser()
sns.set()


class PDBToImageConverter:
    def __init__(
        self,
        data_path,
        pdb_path,
        png_path,
        npy_path,
        csv_path,
        id_list,
        img_dim=None,
        overwrite_files=False,
        selection="protein and heavy",
        pdb_suffix=".pqr",
        channels=["dist", "coulomb", "seqindex"],
        output_format="png",
        save_channels=True,
        randomize_list=False,
        distmat_cutoff=51.2,
        invert_img=False,
        maxnatoms=20000,
        is_pdbbind=True,
        sample_modes=False,
        save_sparse_array=False,
        triangular_matrix=False,
        max_seq_dist=256,
        verbosity=0,
        nprocs=1,
        first_chain=False,
    ):
        self.data_path = data_path
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        self.pdb_path = pdb_path
        if not os.path.exists(pdb_path):
            os.mkdir(pdb_path)
        self.png_path = png_path
        if not os.path.exists(png_path):
            os.mkdir(png_path)
        self.npy_path = npy_path
        if not os.path.exists(npy_path):
            os.mkdir(npy_path)
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            os.mkdir(csv_path)
        self.id_list = id_list
        self.img_dim = img_dim
        self.overwrite_files = overwrite_files
        self.selection = selection
        self.pdb_suffix = pdb_suffix
        self.channels = channels
        if self.channels[0] == "":
            self.channels = []
        self.first_chain = first_chain
        self.output_format = output_format
        self.save_channels = save_channels
        self.distmat_cutoff = distmat_cutoff
        self.invert_img = invert_img
        self.maxnatoms = maxnatoms
        self.is_pdbbind = is_pdbbind
        self.sample_modes = sample_modes
        self.save_sparse_array = save_sparse_array
        self.triangular_matrix = triangular_matrix
        self.max_seq_dist = max_seq_dist
        self.verbosity = verbosity
        self.nprocs = nprocs
        ncores = cpu_count()
        if self.nprocs == -1:
            self.nprocs = ncores - 2
        print("Found %i cores. Using %i." % (ncores, self.nprocs))
        self.canonical_atom_order = {
            "A": ["N", "CA", "C", "O", "CB"],
            "C": ["N", "CA", "C", "O", "CB", "SG"],
            "D": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
            "E": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
            "F": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "G": ["N", "CA", "C", "O"],
            "H": ["N", "CA", "C", "O", "CB", "CG", "CE1", "CD2", "ND1", "NE2"],
            "I": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
            "K": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
            "L": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
            "M": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
            "N": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
            "P": ["N", "CA", "C", "O", "CB", "CG", "CD"],
            "Q": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
            "R": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
            "S": ["N", "CA", "C", "O", "CB", "OG"],
            "T": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
            "V": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
            "W": [
                "N",
                "CA",
                "C",
                "O",
                "CB",
                "CG",
                "CD1",
                "CD2",
                "NE1",
                "CE2",
                "CE3",
                "CZ2",
                "CZ3",
                "CH2",
            ],
            "Y": [
                "N",
                "CA",
                "C",
                "O",
                "CB",
                "CG",
                "CD1",
                "CD2",
                "CE1",
                "CE2",
                "CZ",
                "OH",
            ],
        }
        if randomize_list:
            tmp = len(id_list)
            random.shuffle(id_list)
            assert len(id_list) == tmp and len(id_list) > 0, str(len(id_list))

    def get_anm_crosscorr(self, prot, name="prot"):
        try:
            anm, sel = prody.calcANM(prot, selstr=self.selection)
            if self.sample_modes:
                ens = prody.sampleModes(anm, sel, n_confs=10, rmsd=0.8)
                prody.writePDB(name, ens)
            # print anm,sel
            anm.calcModes()
            cc = prody.calcCrossCorr(anm)
            # prody.view3D(prot, flucts=anm)
            return cc
        except:
            print("ANM failed!")
            return np.array([])

    def save_matrix_npy(self, matrix, label):
        savename = os.path.join(self.npy_path, label + ".npy")
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html
        if not os.path.exists(savename) or self.overwrite_files:
            np.save(savename, matrix, allow_pickle=True)
        return savename

    def load_matrix_np(self, filename):
        return np.load(filename, allow_pickle=True)

    def norm_mat_img(self, matrix):
        if self.invert_img:
            return 256 + (-255 * (matrix - np.amin(matrix)) / np.amax(matrix))
        else:
            return 255 * (matrix - np.amin(matrix)) / np.amax(matrix)

    def save_matrix_1c_png(self, matrix, label):
        savename = os.path.join(self.png_path, label + ".png")
        if not os.path.exists(savename) or self.overwrite_files:
            # mpl.use('Agg')
            matrix = self.norm_mat_img(matrix)
            if self.img_dim != None:
                # print matrix

                matrix = imresize(
                    matrix, (self.img_dim, self.img_dim), interp="nearest", mode="L"
                )
            imsave(savename, matrix)
        return savename

    def load_matrix_1c_png(self, filename):
        return imread(filename, mode="L", flatten=True)

    def load_matrix_3c_png(self, filename):
        return imread(filename, mode="RGB", flatten=False)

    def save_matrix_1c_png_mpl(self, matrix, label):
        savename = os.path.join(self.png_path, label + ".png")
        if not os.path.exists(savename) or self.overwrite_files:
            # mpl.use('Agg')
            matrix = self.norm_mat_img(matrix)
            plt.Figure(figsize=(self.img_dim, self.img_dim), dpi=1)
            ax = plt.axes([0, 0, 1, 1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.imshow(matrix, aspect="equal", origin="lower", interpolation="nearest")
            plt.axis("off")
            plt.savefig(savename, bbox_inches="tight", pad_inches=0, dpi="figure")
            plt.clf()
        return savename

    def make_rgb_tensor(self, r, g, b):
        r = self.norm_mat_img(r)
        g = self.norm_mat_img(g)
        b = self.norm_mat_img(b)

        rgb = np.zeros((r.shape[0], r.shape[1], 3), "uint8")
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    def save_matrix_3c_png(self, rgb, label):
        rgbname = os.path.join(self.png_path, label + ".png")

        imsave(rgbname, rgb)

        return rgbname

    def save_matrix_4c_png(self, label, r, g, b, a):
        rgbaname = os.path.join(self.png_path, label + ".png")
        r = self.norm_mat_img(r)
        g = self.norm_mat_img(g)
        b = self.norm_mat_img(b)
        a = self.norm_mat_img(a)

        rgba = np.zeros((r.shape[0], r.shape[1], 4), "uint8")
        rgba[:, :, 0] = r
        rgba[:, :, 1] = g
        rgba[:, :, 2] = b
        rgba[:, :, 3] = a
        imsave(rgbaname, rgba)
        return rgbaname

    def get_seqdistmat(self, prot):
        n = len(prot)
        resind = prot.getResindices()
        mat = np.zeros((n, n))
        for i in range(n):
            res_i = resind[i]
            for j in range(n):
                res_j = resind[j]
                if i <= j:
                    mat[i][j] = min(self.max_seq_dist, abs(res_i - res_j))
                    mat[j][i] = mat[i][j]
        return mat

    def save_sparse_triangular(self):
        pass

    def pdb2electro(
        self, prot, distmat, mincut=-0.1, maxcut=0.1
    ):  # requires data from PQR file!
        n = len(prot)
        nbmat = np.zeros((n, n))
        charges = prot.getCharges()
        radii = prot.getRadii()
        # pafu = 0
        for i in range(n):
            q_i = charges[i]
            # r_i=radii[i]
            for j in range(n):
                if i < j:
                    q_j = charges[j]
                    # r_j=radii[j]
                    r_ij = distmat[i, j]

                    if r_ij > 0.0:  # and r_ij<=25.6:

                        # https://en.wikipedia.org/wiki/AMBER

                        sig = 12.0  # distance where potential is zero
                        r0_ij = 1.122 * sig  # distance where potential at minimum
                        r6 = math.pow(r0_ij, 6)
                        V_ij = 1.0  # -120.0
                        e_r = 1.0 * math.pi * 4
                        f = 1.0  # 138.935458
                        A_ij = 2 * r6 * V_ij
                        B_ij = (A_ij / 2.0) * r6
                        E_LJ = (-A_ij) / math.pow(r_ij, 6) + (B_ij) / math.pow(r_ij, 12)
                        E_coul = f * (q_i * q_j) / (e_r * r_ij)
                        E_nb = E_LJ + E_coul
                        E_nb = min(maxcut, max(mincut, E_nb))
                        # E_nb = np.exp(-(E_LJ + E_coul)/1.0  )

                        # E_nb = -(E_LJ + E_coul)
                        # if r_ij < 6.0:
                        #    E_nb = -E_nb
                    else:
                        E_nb = 0.0
                        # E_nb = 1.0
                    nbmat[i, j] = E_nb
                    # pafu += E_nb
                    # if i!=j:
                    nbmat[j, i] = E_nb
                    # pafu += E_nb
                    # nbmat = nbmat+np.min(nbmat)
                    # nbmat = nbmat-np.min(nbmat)
                    # nbmat = np.amin(10,nbmat)
                    # nbmat = nbmat/np.max(nbmat)
                    # print np.min(nbmat), np.max(nbmat), np.median(nbmat)
        return nbmat

    def pdb2coulomb(
        self, prot, distmat, mincut=-0.1, maxcut=0.1
    ):  # requires data from PQR file!
        n = len(prot)
        cmat = np.zeros((n, n))
        charges = prot.getCharges()

        for i in range(n):
            q_i = charges[i]
            for j in range(n):

                if i < j:
                    q_j = charges[j]
                    r_ij = distmat[i, j]

                    e_r = 1.0 * math.pi * 4
                    f = 1.0  # 138.935458
                    E_coul = f * (q_i * q_j) / (e_r * r_ij)
                    if E_coul != 0:
                        E_coul = math.log(E_coul)
                    # E_coul = min(maxcut,max(mincut, E_coul ))
                    cmat[i, j] = E_coul

                    cmat[j, i] = E_coul

        return cmat

    """
        Given a protein and pocket object, returns the atom sequence.
        prot: an AtomGroup object defined by the prody package
    """
    def get_atom_sequence(self, prot, pocket):
        d = OrderedDict()
        d["aname"] = prot.getNames()
        # d["elem"]=prot.getElements()
        # d["mass"]=prot.getMasses()
        d["rnum"] = prot.getResnums().astype(np.int16)
        d["altloc"] = prot.getAltlocs()
        # d["rname"]=prot.getResnames()
        d["rseq"] = np.array([s for s in prot.getSequence()])
        d["chid"] = prot.getChids()
        d["chind"] = prot.getChindices()

        # TODO: add charges for .pdb files
        if self.pdb_suffix == ".pqr":
            d["charges"] = prot.getCharges().astype(np.float16)
            d["radii"] = prot.getRadii().astype(np.float16)
        else:
            d["charges"] = [0] * len(d["aname"])
            d["radii"] = [0] * len(d["aname"])
            

        d["x"], d["y"], d["z"] = zip(*prot.getCoords().astype(np.float16))
        d["x"] = np.array(d["x"], dtype=np.float16)
        d["y"] = np.array(d["y"], dtype=np.float16)
        d["z"] = np.array(d["z"], dtype=np.float16)

        if "pocket" in self.channels:
            pocres = np.unique(pocket.getResnums())
            # print(pocres)
            d["pocket"] = np.array(
                [1 if r in pocres else 0 for r in d["rnum"]], dtype=np.int8
            )
        else:
            d["pocket"] = np.zeros(d["rnum"].shape[0])
        # d["coords"] = prot.getCoords()

        df = pd.DataFrame(d)
        df.loc[:, "canonical"] = -1
        print(df.shape)
        # print(df.drop_duplicates().shape)

        if self.first_chain:
            df = df[df["chind"] == 0]
        print(df.shape)
        # sys.exit()
        return df

    def pdb2mat(self, pdb, channels=["dist", "anm", "electro"]):
        if self.pdb_suffix == ".pqr":
            prot = prody.parsePQR(pdb).select(self.selection)
        else:
            prot = prody.parsePDB(pdb).select(self.selection)

        matrices = {}
        distmat = prody.buildDistMatrix(prot)
        for ch in channels:
            if ch == "dist":
                matrices[ch] = distmat
            if ch == "anm":
                ccmap = self.get_anm_crosscorr(
                    prot, name=pdb.strip(self.pdb_suffix) + "_modes.pdb"
                )
                matrices[ch] = ccmap
            if ch == "electro":
                nbmat = self.pdb2electro
                matrices[ch] = nbmat
            if ch == "anmElectro":

                ccmap = self.get_anm_crosscorr(
                    prot, name=pdb.strip(self.pdb_suffix) + "_modes.pdb"
                )
                nbmat = self.pdb2electro
                matrices[ch] = matrices[ch] = np.array(
                    [
                        [
                            ccmap[i, j] if i > j else nbmat[i, j]
                            for j in range(ccmap.shape[0])
                        ]
                        for i in range(ccmap.shape[0])
                    ]
                )
            if ch == "dfk1":
                matrices[ch] = np.power(distmat, -2 * 1)
            if ch == "dfk2":
                matrices[ch] = np.power(distmat, -2 * 2)
            if ch == "dfk3":
                matrices[ch] = np.power(distmat, -2 * 3)
            if ch == "dfk4":
                matrices[ch] = np.power(distmat, -2 * 4)
        return matrices

    def show_protein(dom):
        plt.close("all")
        print("Showing protein")
        q = "pdb:%s" % dom[:4]

        p3d = py3Dmol.view(query=q)
        p3d.setStyle({"cartoon": {"color": "spectrum"}})
        p3d.show()

    # Main function. Does the actual conversion from PDB/PQR file to image/array file.
    def convert_pdbs_to_arrays(
        self,
        verbosity=0,
        show_pairplot=False,
        show_distplots=False,
        show_matrices=False,
        show_protein=False,
    ):
        n = len(self.id_list)

        print("Processing channels:", self.channels)
        if self.pdb_suffix == ".pqr":
            parser = prody.parsePQR
        else:
            parser = prody.parsePDB

        if show_protein:
            import py3Dmol

        print_progress(
            0,
            n,
            prefix="converting structures:",
            suffix="",
            decimals=2,
            bar_length=20,
            prog_symbol="O",
        )

        for di in range(n):
            dom = self.id_list[di]
            if show_protein:
                self.show_protein(dom)

            if self.is_pdbbind:
                p = os.path.join(
                    self.pdb_path, dom, "%s_protein%s" % (dom, self.pdb_suffix)
                )
                # assert os.path.exists(p)
                pp = os.path.join(
                    self.pdb_path, dom, "%s_pocket%s" % (dom, ".pdb")
                )  ## currently not using PQR for pockets!!!!
                if not os.path.exists(pp):
                    if verbosity > 0:
                        print("Error: could not find ", pp)
                    continue
            else:
                p = os.path.join(self.pdb_path, "%s%s" % (dom, self.pdb_suffix))
                pp = None

            if not os.path.exists(p):
                if verbosity > 0:
                    print("Error: could not find ", p)
                continue                

            ## png image things
            for_sparse = []
            matrices = {}
            distmat = np.array([])
            # dm_I = np.array([])
            # dm_mask = np.array([])
            prot = None
            nat = 0
            rgblabel = dom + "_rgb"
            if self.output_format == "npy":
                rgbfile = os.path.join(self.npy_path, rgblabel + ".npy")
            else:
                rgbfile = os.path.join(self.png_path, rgblabel + ".png")

            if not self.overwrite_files:
                is_missing = {}
                for ch in self.channels:
                    if self.output_format == "png":
                        tmppath = os.path.join(self.png_path, dom + "_%s.png" % ch)
                    elif self.output_format == "npy":
                        tmppath = os.path.join(self.npy_path, dom + "_%s.npy" % ch)
                    else:
                        tmppath = None

                    if not os.path.exists(tmppath):
                        is_missing[ch] = True
                    else:
                        is_missing[ch] = False
                # if all files are there, no need to continue
                if not any(
                    [is_missing[k] for k in is_missing.keys()]
                ) and os.path.exists(rgbfile):
                    continue

            distmat = None
            coulombmat = None
            seqdistmat = None
            seqindmat = None
            distmat_pocket = None

            try:
                if self.verbosity > 1:
                    print("Parse protein:")

                prot = parser(p).select(self.selection, quiet=True)

                pocket = None
                if self.is_pdbbind:
                    if self.verbosity > 1:
                        print("Parse pocket:")
                    pocket = prody.parsePDB(pp).select(self.selection, quiet=False)
                    if self.verbosity > 1:
                        print("pocket atoms:", pocket.numAtoms())
            except Exception as e:
                print("ERROR while parsing %s:" % p, e)
                continue
            # EXTRACT ATOM SEQUENCE
            atom_df = self.get_atom_sequence(prot, pocket)

            nat = prot.numAtoms()
            if nat > self.maxnatoms:
                prot = None
                print(
                    "skipping %s: exceeds self.maxnatoms=%i. " % (dom, self.maxnatoms),
                    nat,
                )
                continue

            if show_protein:
                print("prody.showProtein(prot)")
                plt.close("all")
                prody.showProtein(prot)
                plt.show()

            # RUN CHECKS ON ATOM_DF, ESPECIALLY ATOM ORDER PER RESIDUE
            print_progress(
                di + 1,
                n,
                prefix="converting structures:",
                suffix=" - %s - %i atoms - calculating: %s"
                % (dom, nat, "atom csv".ljust(12)),
                decimals=2,
                bar_length=20,
                prog_symbol="O",
            )

            if self.verbosity > 1:
                print("\n" + p)
                print(atom_df.info())
                print(atom_df.head())
                print(atom_df.tail())
                print(prot.numAtoms())
                # assert atom_df.shape[0] == prot.numAtoms()

            chains = atom_df["chid"].unique()
            chains_i = atom_df["chind"].unique()

            if self.verbosity > 1:
                print("chains:", chains, chains_i)
                assert len(chains) == len(chains_i)
            try:
                for chain in chains:
                    unires = atom_df[atom_df["chid"] == chain].loc[:, "rnum"].unique()
                    if self.verbosity > 1:
                        print("residues:", unires)
                    for r in unires:
                        alts = (
                            atom_df[(atom_df["rnum"] == r) & (atom_df["chid"] == chain)]
                            .loc[:, "altloc"]
                            .unique()
                        )
                        if len(alts) > 1:
                            if self.verbosity > 1:
                                print("altloc:", alts)
                        aa = (
                            atom_df[(atom_df["rnum"] == r) & (atom_df["chid"] == chain)]
                            .loc[:, "rseq"]
                            .unique()
                        )
                        assert len(aa) == 1, str(
                            atom_df[(atom_df["rnum"] == r) & (atom_df["chid"] == chain)]
                        )

                        aa = aa[0]
                        alist = list(
                            atom_df[
                                (atom_df["rnum"] == r) & (atom_df["chid"] == chain)
                            ].loc[:, "aname"]
                        )
                        can = self.canonical_atom_order[aa]
                        if self.verbosity > 1:
                            if alist != can:

                                print(r, chain, aa, alist, can)
                                if "OXT" in alist:
                                    print("Found terminal OXT")
                                elif sorted(alist) == sorted(can):
                                    print("non-canonical atom order")

                                else:
                                    pass

                                    # sys.exit()

                                # print("canonical=0")
                            else:
                                pass
                                # print("canonical=1")
                        atom_df.loc[
                            (atom_df["rnum"] == r) & (atom_df["chid"] == chain),
                            "canonical",
                        ] = (alist == can)
                        # print(atom_df[ (atom_df["rnum"]==r) & (atom_df["chid"]==chain) ])

                csvname = os.path.join(
                    self.data_path, self.csv_path, dom + "_atomsequence.csv"
                )
                if self.verbosity >= 1:
                    print("Writing: ", csvname)
                atom_df.to_csv(csvname, index_label="index")
            except Exception as e:
                print("ERROR while checking atoms in %s:" % p, e)
                continue

            if len(self.channels) == 0:
                continue

            # BUILD DISTANCE BASED MATRICES
            if self.verbosity > 1:
                print("build dist matrices...")
            if self.nprocs > 1:
                (
                    distmat,
                    coulombmat,
                    seqdistmat,
                    seqindmat,
                    distmat_pocket,
                ) = self.my_mats_p(atom_df)
            else:
                (
                    distmat,
                    coulombmat,
                    seqdistmat,
                    seqindmat,
                    distmat_pocket,
                ) = self.my_mats(atom_df)

            for ch in self.channels:
                if self.overwrite_files or is_missing[ch]:
                    print_progress(
                        di + 1,
                        n,
                        prefix="converting structures:",
                        suffix=" - %s - %i atoms - calculating: %s"
                        % (dom, nat, ch.ljust(12)),
                        decimals=2,
                        bar_length=20,
                        prog_symbol="O",
                    )

                    if ch == "dist":
                        matrices[ch] = distmat
                    elif ch == "pocket":
                        matrices[ch] = distmat_pocket
                    elif ch == "coulomb":
                        matrices[ch] = coulombmat
                    elif ch == "seqdist":
                        matrices[ch] = seqdistmat
                    elif ch == "seqindex":
                        matrices[ch] = seqindmat
                    else:
                        print("Unknown channel:", ch)
                        continue
                        # sys.exit(1)

                    if show_distplots:
                        plt.close("all")
                        histo = sns.distplot(
                            matrices[ch].flatten(), bins=256, kde=False, color="r"
                        )
                        # histo.set_yscale('log')
                        plt.xlabel("normalized values")
                        plt.ylabel("counts")
                        plt.title("matrix histogram: " + ch)
                        plt.show()

                    if nat > 0 and self.save_channels:

                        if self.output_format == "npy":
                            savename = self.save_matrix_npy(
                                matrices[ch], dom + "_" + ch
                            )
                        else:
                            savename = self.save_matrix_1c_png(
                                matrices[ch], dom + "_" + ch
                            )
                        assert os.path.exists(savename)
                else:

                    if verbosity > 0:
                        print("loading", ch)
                    if self.output_format == "npy":
                        matrices[ch] = self.load_matrix_np(
                            os.path.join(self.npy_path, dom + "_%s.npy" % ch)
                        )

                    else:
                        matrices[ch] = self.load_matrix_1c_png(
                            os.path.join(self.png_path, dom + "_%s.png" % ch)
                        )
                    print_progress(
                        di + 1,
                        n,
                        prefix="converting structures:",
                        suffix=" - %s - %i atoms - from file: %s"
                        % (dom, matrices[ch].shape[0], ch.ljust(12)),
                        decimals=2,
                        bar_length=20,
                        prog_symbol="O",
                    )

                if show_matrices:
                    plt.close("all")
                    cax = plt.imshow(matrices[ch], interpolation="none")
                    plt.gca().grid(False)
                    plt.colorbar(cax)
                    plt.title("atomic pairwise matrix (%s); %s" % (ch, self.selection))
                    plt.show()
            if len(matrices) >= 3:
                if show_pairplot:

                    tmpdf = pd.DataFrame(
                        data={ch: matrices[ch].flatten() for ch in self.channels[:3]}
                    )
                    pairplot_n = len(tmpdf.index) / 10
                    sns.pairplot(tmpdf.ix[random.sample(tmpdf.index, pairplot_n)])
                    plt.title("%i samples from 2D matrix" % (pairplot_n))
                    plt.show()
                print_progress(
                    di + 1,
                    n,
                    prefix="converting structures:",
                    suffix=" - %s - %i atoms - writing: %s"
                    % (dom, matrices[ch].shape[0], "RGB".ljust(12)),
                    decimals=2,
                    bar_length=20,
                    prog_symbol="O",
                )

                rgb = self.make_rgb_tensor(
                    matrices[self.channels[0]],
                    matrices[self.channels[1]],
                    matrices[self.channels[2]],
                )
                if show_matrices:
                    plt.imshow(rgb, interpolation="none")
                    plt.gca().grid(False)
                    plt.title("atomic pairwise matrix (RGB); %s" % (self.selection))
                    plt.show()

                if self.overwrite_files or not os.path.exists(rgbfile):
                    if not self.save_sparse_array:
                        if self.output_format == "npy":
                            savename = self.save_matrix_npy(rgb, rgblabel)
                        else:
                            savename = self.save_matrix_3c_png(rgb, rgblabel)
                        assert os.path.exists(savename)
                    else:
                        for_sparse.append(matrices)

    def my_mats_p(self, atom_df, mincut=-0.1, maxcut=0.1):
        n = len(atom_df["aname"])
        distmat = np.zeros((n, n))
        coulomb = np.zeros((n, n))
        seqdist = np.zeros((n, n))
        seqindex = np.zeros((n, n))
        distmat_pocket = np.zeros((n, n))

        pool = Pool(processes=self.nprocs)

        parms = {
            "do_pocket": "pocket" in self.channels,
            "e_r": 1.0 * math.pi * 4,
            "f": 1.0,
            "max_seq_dist": self.max_seq_dist,
            "distmat_cutoff": self.distmat_cutoff,
            "cmincut": -0.5,
            "cmaxcut": 0.05,
        }

        parr = (
            atom_df["pocket"].values
            if parms["do_pocket"]
            else np.zeros(atom_df.shape[0])
        )

        coords = np.array(list(zip(atom_df["x"], atom_df["y"], atom_df["z"])))

        if self.verbosity >= 1:
            print("parallel calculation across ", self.nprocs)
            start = time.time()

        results = pool.map(
            par.proxy,
            par.data_stream_val(
                range(n),
                coords,
                atom_df["charges"].values,
                atom_df["rnum"].values,
                parr,
                parms,
            ),
            chunksize=1000,
        )

        if self.verbosity >= 1:
            end = time.time()
            elapsed = end - start
            print(
                "Elapsed time after calculating matrix elements (s): %s" % str(elapsed)
            )

            start = time.time()
        for k, v in results:
            i, j = k
            dmat, cmat, sdmat, simat, pmat = v
            # print (i,j, dmat, cmat, sdmat, simat, pmat)
            distmat[i, j] = distmat[j, i] = dmat
            coulomb[i, j] = coulomb[j, i] = cmat
            seqdist[i, j] = seqdist[j, i] = sdmat
            seqindex[i, j] = seqindex[j, i] = simat
            if parms["do_pocket"]:
                distmat_pocket[i, j] = distmat_pocket[j, i] = pmat

        if self.verbosity >= 1:
            end = time.time()
            elapsed = end - start
            print("Elapsed time after writing matrices (s): %s" % str(elapsed))

        return distmat, coulomb, seqdist, seqindex, distmat_pocket

    def my_mats(self, atom_df, mincut=-0.1, maxcut=0.1):
        n = len(atom_df["aname"])
        distmat = np.zeros((n, n))
        coulomb = np.zeros((n, n))
        seqdist = np.zeros((n, n))
        seqindex = np.zeros((n, n))

        do_pocket = "pocket" in atom_df
        if do_pocket:
            distmat_pocket = np.zeros((n, n))
        else:
            distmat_pocket = None

        for a1 in range(n):
            xyz1 = np.array(
                (atom_df.loc[a1, "x"], atom_df.loc[a1, "y"], atom_df.loc[a1, "z"])
            )  # atomseq["coords"][a1]

            q1 = atom_df.loc[a1, "charges"]

            r1 = atom_df.loc[a1, "rnum"]
            for a2 in range(n):
                if a1 < a2:
                    xyz2 = np.array(
                        (
                            atom_df.loc[a2, "x"],
                            atom_df.loc[a2, "y"],
                            atom_df.loc[a2, "z"],
                        )
                    )  # atomseq["coords"][a2]
                    q2 = atom_df.loc[a2, "charges"]
                    r2 = atom_df.loc[a2, "rnum"]
                    d = prody.measure.measure.getDistance(xyz1, xyz2)
                    r_ij = d
                    e_r = 1.0 * math.pi * 4
                    f = 1.0
                    E_coul = f * (q1 * q2) / (e_r * r_ij)
                    # if E_coul > 0:
                    #    E_coul = math.log(E_coul)/100.0
                    # elif E_coul < 0:
                    #    E_coul = -math.log(-E_coul)/100.0
                    # E_coul = min(maxcut,max(mincut, E_coul ))
                    distmat[a1, a2] = distmat[a2, a1] = d
                    coulomb[a1, a2] = coulomb[a2, a1] = E_coul
                    seqdist[a1, a2] = seqdist[a2, a1] = min(
                        self.max_seq_dist, abs(r1 - r2)
                    )
                    seqindex[a1, a2] = seqindex[a2, a1] = (
                        min(r1, r2) % 2 * min(self.max_seq_dist, abs(r1 - r2))
                    )
                    if do_pocket:
                        distmat_pocket[a1, a2] = distmat_pocket[a2, a1] = (
                            atom_df.loc[a1, "pocket"] and atom_df.loc[a2, "pocket"]
                        )
                elif a1 == a2:
                    distmat[a1, a2] = atom_df.loc[a1, "radii"]
                    coulomb[a1, a2] = q1
        # coulomb = (coulomb-mincut)/maxcut
        return distmat, coulomb, seqdist, seqindex, distmat_pocket


def overlapping_tiles_from_images(
    img_folder,
    tiles_folder,
    tile_size=512,
    n_channels=3,
    suffix="_rgb.png",
    show_plots=False,
    write_individual_tiles=False,
):
    glob_string = os.path.join(img_folder, "*%s" % (suffix))
    print(glob_string)
    files = glob.glob(glob_string)
    print("%i files found" % len(files))

    # h5_img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved

    for f in files[:1]:
        identifier = os.path.basename(f).split(".")[0]

        if n_channels == 3:
            img = imread(f, flatten=False, mode="RGB")
        else:
            img = imread(f, flatten=True)

        if n_channels == 1:
            x = np.array(img)[..., np.newaxis]
        else:
            x = np.array(img)

        print(identifier, x.shape)

        dim = x.shape[0]
        if show_plots:
            plt.imshow(x, interpolation="none")
            plt.gca().grid(False)
            plt.title("full image")
            plt.show()

        n_splits = math.ceil(dim / float(tile_size))
        start = math.floor(dim / n_splits)

        tiles_per_image = []
        hdf5_path = os.path.join(tiles_folder, "%s_tile%i.h5" % (identifier, tile_size))

        coords = []
        counter = 0
        for i in range(n_splits):
            for j in range(n_splits):
                # print(i,j, n_splits, start)
                xij = np.zeros((tile_size, tile_size, n_channels), dtype=np.int)

                tile = x[
                    i * start : i * start + tile_size,
                    j * start : j * start + tile_size,
                    :,
                ]

                xij[: tile.shape[0], : tile.shape[1], :] = tile

                # print(xij.shape)

                if show_plots:
                    plt.imshow(xij, interpolation="none")
                    plt.gca().grid(False)
                    plt.title("%i,%i" % (i, j))
                    plt.show()
                coords.append((i * start, j * start))
                tiles_per_image.append(xij)
                if write_individual_tiles:
                    imsave(
                        os.path.join(
                            tiles_folder,
                            "%s_tile%i_%i_%i_%i%s"
                            % (
                                identifier,
                                tile_size,
                                counter,
                                i * start,
                                j * start,
                                suffix,
                            ),
                        ),
                        xij,
                    )
                counter += 1
        hdf5_file = tables.open_file(hdf5_path, mode="w")
        a = np.array(tiles_per_image, dtype=np.uint8)
        print(a.shape)
        hdf5_file.create_array(hdf5_file.root, "tiles", a)
        hdf5_file.create_array(hdf5_file.root, "xy", coords)
        hdf5_file.close()


# adapted from ProDy
def buildDistMatrixAnnot(
    atoms1, atoms2=None, annot_group=None, unitcell=None, format="mat"
):
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
    annot_resnums = annot_group.getResnums()
    annot_chains = annot_group.getChids()

    if " " in annot_chains:  # WHAT DOES THIS MEAN?
        unique_chains = pd.Series(prot_atoms.getChids()).unique()
        annot_chains = [unique_chains[0] for _ in annot_chains]

    annot_rc_pairs = zip(annot_resnums, annot_chains)
    atoms1_is_pocket = np.zeros(len(atoms1))
    for i, xyz_i in enumerate(atoms1):
        prot_res_num_i = prot_atoms[i].getResnum()
        prot_res_chain_i = prot_atoms[i].getChid()
        if (prot_res_num_i, prot_res_chain_i) in annot_rc_pairs:
            atoms1_is_pocket[i] = 1

    if not isinstance(atoms1, np.ndarray):
        try:
            atoms1 = atoms1._getCoords()
        except AttributeError:
            raise TypeError("atoms1 must be Atomic instance or an array")
    if atoms2 is None:
        symmetric = True
        atoms2 = atoms1
    else:
        symmetric = False
        if not isinstance(atoms2, np.ndarray):
            try:
                atoms2 = atoms2._getCoords()
            except AttributeError:
                raise TypeError("atoms2 must be Atomic instance or an array")
    if atoms1.shape[-1] != 3 or atoms2.shape[-1] != 3:
        raise ValueError("one and two must have shape ([M,]N,3)")

    if unitcell is not None:
        if not isinstance(unitcell, np.ndarray):
            raise TypeError("unitcell must be an array")
        elif unitcell.shape != (3,):
            raise ValueError("unitcell.shape must be (3,)")

    dist = np.zeros((len(atoms1), len(atoms2)))
    dist_annot = np.zeros((len(atoms1), len(atoms2)))
    if symmetric:
        if format not in prody.measure.measure.DISTMAT_FORMATS:
            raise ValueError("format must be one of mat, rcd, or arr")
        if format == "mat":
            for i, xyz_i in enumerate(atoms1):

                for j, xyz_j in enumerate(atoms2):

                    if i > j:

                        dist[i, j] = dist[j, i] = prody.measure.measure.getDistance(
                            xyz_i, xyz_j, unitcell
                        )
                        if atoms1_is_pocket[i] and atoms1_is_pocket[j]:

                            dist_annot[i, j] = dist_annot[j, i] = dist[i, j]

        else:
            dist = np.concatenate(
                [
                    prody.measure.measure.getDistance(xyz, atoms2[i + 1 :], unitcell)
                    for i, xyz in enumerate(atoms1)
                ]
            )
            if format == "rcd":
                n_atoms = len(atoms1)
                rc = np.array(
                    [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
                )
                row, col = rc.T
                dist = (row, col, dist)

    else:
        for i, xyz in enumerate(atoms1):
            dist[i] = prody.measure.measure.getDistance(xyz, atoms2, unitcell)
    return dist, dist_annot


#  Obtain domain list from file. Domain IDs are the first white-space separated entry in a line. Comments (#...) are ignored.
def getDomList(loc):
    assert os.path.exists(loc), "{} does not exist".format(loc)
    with open(loc, "r") as domlistfile:
        lines = domlistfile.readlines()  # .split()
    idlist = []
    for l in lines:
        line = l  # .decode("utf-8-sig")
        linesplit = line.split()
        if line[0] != "#" and len(linesplit) > 0:
            idlist.append(linesplit[0])
    return idlist


# Run external executable of "pdb2pqr" on PDB files from a domain list within a specified folder and converts them to PQR files in a different folder.
def pdb2pqr(
    idlist,
    pdbloc,
    pqrloc,
    exe="pdb2pqr",
    show_output=False,
    show_error=False,
    ff="amber",
    quiet=False,
    is_pdbbind=False,
    options=" --chain   " % (),
    overwrite=False,
):
    # DOC: https://pdb2pqr.readthedocs.io/en/latest/getting.html
    # binaries: https://sourceforge.net/projects/pdb2pqr/
    assert os.path.exists(
        exe
    ), "Could not find PDB2PQR executable, please check path:" + os.path.abspath(exe)

    if not quiet:
        print_progress(0, len(idlist), prefix="Convert PDB to PQR")
    for di in range(len(idlist)):
        dom = idlist[di]
        if is_pdbbind:
            dompdb = os.path.join(pdbloc, dom, "%s_protein.pdb" % dom)
            dompqr = os.path.join(pdbloc, dom, "%s_protein.pqr" % dom)
            ligand = ' --ligand="%s" ' % (
                os.path.join(pdbloc, dom, "%s_ligand.mol2" % dom)
            )
        else:
            dompdb = os.path.join(pdbloc, "%s.pdb" % dom[:4].lower() + dom[4:].upper())
            dompqr = os.path.join(pqrloc, "%s.pqr" % dom)
            ligand = ""
        if not quiet:
            print(dompdb, dompqr, os.path.exists(dompdb) and not os.path.exists(dompqr))
        if os.path.exists(dompdb) and (not os.path.exists(dompqr) or overwrite):
            cmd_string = "%s %s --ff=%s %s %s %s" % (
                exe,
                options,
                ff,
                ligand,
                dompdb,
                dompqr,
            )
            print(cmd_string)
            p = sub.Popen(cmd_string, shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
            err = p.stderr.read()
            out = p.stdout.read()
            if show_output:
                print("<OUTPUT %s>" % dompdb, out)
            if show_error and err.strip() != "":
                print("<ERROR %s>" % dompdb, err)

            p.wait()
        else:
            if not quiet:
                print(
                    " - %s exists? %i, %s exists? %i"
                    % (
                        dompdb,
                        int(os.path.exists(dompdb)),
                        dompqr,
                        int(os.path.exists(dompqr)),
                    )
                )
        try:
            if not quiet:
                print_progress(di, len(idlist), prefix="Convert PDB to PQR", suffix=dom)
        except UnicodeDecodeError as ude:
            print(di)
            print(dom)
            print(ude)


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
                assert len(tmp) == 2
                d["occupancy"] = float(tmp[0])
                d["bfactor"] = float(tmp[1])
            else:
                d["occupancy"] = float(line[54:60])
                d["bfactor"] = float(line[60:66])
                if len(line) >= 66:
                    d["segment"] = line[72:76].strip()
                    d["element"] = line[76:78].strip()
                    d["charge"] = line[78:80].strip()
    except ValueError as ve:
        print(ve)
        print(line)
        sys.exit(1)
    if returnDict:
        return d
    else:
        if is_pqr:
            return (
                d["ltype"],
                d["anum"],
                d["aname"],
                d["altloc"],
                d["rname"],
                d["chain"],
                d["rnum"],
                d["insert"],
                d["x"],
                d["y"],
                d["z"],
                d["occupancy"],
                d["bfactor"],
            )
        else:
            return (
                d["ltype"],
                d["anum"],
                d["aname"],
                d["altloc"],
                d["rname"],
                d["chain"],
                d["rnum"],
                d["insert"],
                d["x"],
                d["y"],
                d["z"],
                d["occupancy"],
                d["bfactor"],
                d["segment"],
                d["element"],
                d["charge"],
            )


# Converts any PDB/PQR file to a pandas data frame. Note that for PQR, "occupancy" is the partial atomic charge and "bfactor" is the atom radius, and these are the final two entries in each ATOM row.
def pdb_to_dataframe(
    pdbfilename,
    col_labels=[
        "ltype",
        "anum",
        "aname",
        "altloc",
        "rname",
        "chain",
        "rnum",
        "insert",
        "x",
        "y",
        "z",
        "occupancy",
        "bfactor",
        "segment",
        "element",
        "charge",
    ],
    is_pqr=False,
):

    atomlines = []
    with open(pdbfilename) as pdbfile:
        lines = pdbfile.readlines()

        for l in lines:
            if l[:6] == "ATOM  ":
                atomlines.append(
                    list(parseATOMline(l, returnDict=False, is_pqr=is_pqr))
                )
    atomlines = np.array(atomlines).T
    if is_pqr:
        col_labels = col_labels[:13]
    data = {col_labels[li]: atomlines[li] for li in range(len(col_labels))}
    df = pd.DataFrame(data=data, columns=col_labels)

    return df


# Go through all PDB/PQR files in the folder and check for consistent atom lists per residue.
# Outputs a dictionary ("atomchecker") of resiude names in 3-letter code mapped to dictoinary mapping a specific atom name tuple ( order as found in file) to number of observations
def check_atom_order_pdbs(path, pdb_suffix, idlength=7, is_pqr=False):
    fp = os.path.join(path, "%s%s" % (idlength * "?", pdb_suffix))
    AA_3_to_1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    AAs = AA_3_to_1.keys()
    files = glob.glob(fp)
    print(fp, len(files))
    atomchecker = {aa: {} for aa in AAs}  # list of atom-order tuple - frequency pairs
    print_progress(0, len(files))
    for pi in range(len(files[:10])):
        print_progress(pi, len(files))
        p = files[pi]
        # print pi, p
        # lines=[]
        df = pdb_to_dataframe(p, is_pqr=is_pqr)
        # print df
        rnums = df["rnum"].unique()
        for r in rnums:
            inserts = df[df["rnum"] == r]["insert"].unique()
            # TODO:
            rname = df[(df["rnum"] == r) & (df["insert"] == inserts[0])][
                "rname"
            ].unique()
            altlocs = df[df["rnum"] == r]["altloc"].unique()

            if len(rname) != 1:
                print(
                    "%s >> WARNING: multiple rnames per rnum %s! %s (altlocs: %s; inserts: %s)"
                    % (p, r, str(rname), str(altlocs), str(inserts))
                )
            rname = rname[0]
            aorder = str(",".join(df[df["rnum"] == r]["aname"]))
            # print r, rname
            if rname in atomchecker:
                if aorder in atomchecker[rname]:
                    atomchecker[rname][aorder] += 1
                else:
                    atomchecker[rname][aorder] = 1
            else:
                print("Found non-standard AA:", rname)
                atomchecker[rname] = {}
                atomchecker[rname][aorder] = 1

    return atomchecker


if __name__ == "__main__":
    # import cProfile
    # # if check avoids hackery when not profiling
    # # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
    # if sys.modules['__main__'].__file__ == cProfile.__file__:
    #     import prody  # Imports you again (does *not* use cache or execute as __main__)
    #     globals().update(vars(prody))  # Replaces current contents with newly imported stuff
    #     sys.modules['__main__'] = prody  # Ensures pickle lookups on __main__ find matching version

    userid="none"
    
    # PARSE COMMAND LINE ARGUMENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--basepath", default=f"/home/{userid}/data", help="")
    ap.add_argument(
        "--idlistfile", default="cath-dataset-nonredundant-S40.list.txt", help="List of ID's to parse"
    )
    ap.add_argument(
        "--pdbpath",
        default="dompdb",
        help="PDB files containing protein structure data",
    )
    ap.add_argument(
        "--pqrpath", default="dompqr", help="To store PQR files converted from PDB"
    )
    ap.add_argument(
        "--pngpath", default="dompng", help="To store images of protein structures"
    )
    ap.add_argument(
        "--csvpath", default="csv", help="To store data frames with atom sequence"
    )
    ap.add_argument("--h5name", default="h5out.h5", help="")
    ap.add_argument("--imgdim", type=int, default=None, help="")
    ap.add_argument(
        "--selection",
        default="protein and aminoacid and heavy and not hetero and not nucleic and not water",
        help="a valid ProDy selection",
    )
    ap.add_argument("--overwrite", action="store_true", default=False, help="")
    ap.add_argument("--calcpqr", action="store_true", default=False, help="Option to convert PDB files to PQR")
    ap.add_argument(
        "--informat", choices=["pdb", "pqr"], default="pqr", help="pdb or pqr"
    )
    ap.add_argument(
        "--outformat",
        choices=["png", "npy"],
        default="png",
        help="png or npy - beware that npy binary files are huge!",
    )
    ap.add_argument(
        "--pdb2pqr",
        default=f"/home/{userid}/pdb2pqr-linux-bin64-2.1.1/pdb2pqr",
        help="path to the pdb2pqr executable",
    )
    ap.add_argument(
        "--savechannels",
        type=bool,
        default=True,
        help="If three channels are provided, an rgb output will be written. Do you want to save the individual channels as well?",
    )
    ap.add_argument(
        "--randomize", action="store_true", help="randomize the order of inputs"
    )
    ap.add_argument("--distmat_cutoff", type=float, default=51.2, help="")
    ap.add_argument("--show_plots", action="store_true", help="")
    ap.add_argument("--invert_img", action="store_true", help="")
    ap.add_argument("--check_order", action="store_true", help="")
    ap.add_argument("--maxnatoms", type=int, default=20000, help="")
    ap.add_argument("--pdbbind", action="store_true", help="")
    ap.add_argument("--channels", default="dist,coulomb,seqindex", help="")
    ap.add_argument("--verbosity", type=int, default=0, help="")
    ap.add_argument("--sample", type=int, default=-1, help="if >=0, sample from idlist")
    ap.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help="Number of processes to use in parallel. Set to -1 to use all cores minus 2.",
    )
    ap.add_argument("--download_files", action="store_true", help="Downloads all PDB files indicated in ID list, if not already downloaded")

    args = ap.parse_args()
    print("Settings:", args)

    # obtain list of all domains to include in the data set from file
    if args.pdbbind:
        # import encodings
        meta_df = pd.read_csv(os.path.join(args.basepath, args.idlistfile))
        idlist = list(meta_df[meta_df["is_refined"] == True]["PDB code"].values)
    else:
        idlist = getDomList(os.path.join(args.basepath, args.idlistfile))

    print("{} PDB files expected".format(len(idlist)))

    if args.sample >= 0:
        print("Drawing %i id samples." % args.sample)
        idlist = random.sample(idlist, args.sample)

    # If PDB files not available, download them using Prody
    if args.download_files:
        for item in idlist:
            filename = "{}.{}".format(item, "pdb")
            path = os.path.join(args.basepath, args.pdbpath, filename)
            if not os.path.exists(path):
                atoms = prody.parsePDB(item)
                prody.writePDB(path, atoms)

    # optionally, if no PQR files available, convert a folder of PDB files to a folder of PQR files. PQR needed when partial charges required.
    if args.calcpqr:
        print(
            "converting PDB files at %s to PQR files at %s"
            % (args.pdbpath, args.pqrpath)
        )
        pdb2pqr(
            idlist,
            os.path.join(args.basepath, args.pdbpath),
            os.path.join(args.basepath, args.pqrpath),
            exe=args.pdb2pqr,
            is_pdbbind=args.pdbbind,
            show_output=True,
            show_error=True,
            quiet=False,
            overwrite=args.overwrite,
        )

    if args.informat == "pdb":
        suffix = ".pdb"
        pdbpath = args.pdbpath
    elif args.informat == "pqr":
        if args.pdbbind:
            pdbpath = args.pdbpath
        else:
            pdbpath = args.pqrpath
        suffix = ".pqr"
    else:
        print("unknown input format")
        sys.exit(1)

    # control ProDy's output
    # http://prody.csb.pitt.edu/manual/reference/prody.html#module-prody
    prody.confProDy(
        verbosity="critical"
    )  # (‘critical’, ‘debug’, ‘error’, ‘info’, ‘none’, or ‘warning’)

    # initialize object with all arguments
    converter = PDBToImageConverter(
        args.basepath,
        os.path.join(args.basepath, pdbpath),
        os.path.join(args.basepath, args.pngpath),
        os.path.join(args.basepath, args.h5name),
        os.path.join(args.basepath, args.csvpath),
        idlist,
        img_dim=args.imgdim,
        overwrite_files=args.overwrite,
        selection=args.selection,
        pdb_suffix=suffix,
        channels=args.channels.split(","),
        # channels=["anmElectro"],
        output_format=args.outformat,
        save_channels=args.savechannels,
        randomize_list=True,
        distmat_cutoff=args.distmat_cutoff,
        invert_img=args.invert_img,
        maxnatoms=args.maxnatoms,
        is_pdbbind=args.pdbbind,
        verbosity=args.verbosity,
        nprocs=args.nprocs,
    )

    # Go through all PDB/PQR files in the folder and check for consistent atom lists per residue.
    # Outputs a dictionary ("atomchecker") of resiude names in 3-letter code mapped to dictoinary mapping a specific atom name tuple ( order as found in file) to number of observations
    # if args.check_order:
    #     atomchecker= check_atom_order_pdbs(os.path.join(args.basepath, pdbpath),suffix,idlength=4, is_pqr=True if args.informat=="pqr" else False)

    #     sorted_achecker = {}
    #     # print atomchecker to screen
    #     for aa in sorted(atomchecker.keys()):
    #         print( aa )
    #         aos = sorted(atomchecker[aa].items(), key=operator.itemgetter(1), reverse=True )
    #         sorted_achecker[aa] = aos
    #         for aord in aos:
    #             print( "\t%s:\t %s"%(str(aord[1]) , str(aord[0])) )
    #     # write atomchecker dictionary to json file
    #     with open('atomchecker.json', 'w') as fp:
    #         json.dump(sorted_achecker, fp, indent=4, sort_keys=True)
    # else:
    #     # convert PDB/PQR files to image / numpy array
    converter.convert_pdbs_to_arrays(
        show_pairplot=False,
        show_distplots=args.show_plots,
        show_matrices=args.show_plots,
        show_protein=args.show_plots,
        verbosity=args.verbosity,
    )

    # TIP:
    # Split text file into n equal chunks in bash to run in parallel:
    # split -n 5 textfile.txt chunktxt
