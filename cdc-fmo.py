from __future__ import print_function
import numpy as np
import numpy.linalg as LA
import scipy.sparse
import pandas as pd
import matplotlib as plt
import subprocess
import argparse
import logging
import sys

def warnings(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description=
"""
Program for computing Charge Density Coupling (cdc) disaggregated by location.

note: 
UBIF == undefined behavior if false
 EIF == error if false

INPUT
==========================
where       |   -groframe filename
format      |   .gro file format
restriction |   UBIF: Single-frame gromacs file

where       |   -qtop filename
description |   Charge-only pseudo topology
format      |   same as .top format, sans headings, but with all atoms 
            |       verbosely enumerated
restriction |   UBIF: < 100,000 non-solvent atoms
            |   UBIF: Ordered as NON-SOLVENT, SOLVENT, EOF
            |   UBIF: Molecule for cdc is the final nonsolvent molecule

where       |   -qcdc filename
description |   cdc charges in tabular format from original Ceccarelli 
            |       publication
restriction |   EIF: All atom names in -qcdc are present in the molecule
            |       found from -cdc_molname matches in the -groframe

where       |   -cdc_molname string -cdc_molnum int
description |   Find all atoms in the .gro file with -cdc_molname, and split
            |   them into -cdc_molnum evenly sized groups
restriction |   EIF: Atoms of type -cdc_molname must be able to be split into 
            |       evenly sized groups

OUTPUT
==========================
where       |   stdout
format      |   for t in xrange(time):
            |       for i in xrange(sites):
            |           print(" ".join([str(U) for U in U_group[i]]))
description |   Index [0,-1] are grouped by residue 
            |   Index [-1] is all solvent atoms (?)

""")
    parser.add_argument('-groframe', required=True)
    parser.add_argument('-qtop', required=True)
    parser.add_argument('-qcdc', required=True)
    parser.add_argument('-cdc_molname', type=str, default='BCL')
    parser.add_argument('-cdc_molnum', type=int, default=7)
    parser.add_argument('-nosolv', action='store_true', help="Exclude the solvent group")
    parser.add_argument('-total', action='store_true', help="Print only the aggregated sum")
    parser.add_argument('-res', type=int, help="Print only the selected residue (one-based index)")
    parser.add_argument('-debug', action='store_true')
    args = parser.parse_args()
    fname_trj=args.groframe
    fname_top=args.qtop
    fname_cdc=args.qcdc
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.total and (not args.res is None):
        raise ValueError("-total and -res are both set, please use only one of those options")

    N_framelines = 0
    str_box      = ""
    # Load N_framelines and str_box
    with open(fname_trj) as f:
        f.readline()
        N_framelines = int(f.readline().strip())
    p = subprocess.Popen(["tail", "-n1", fname_trj], stdout=subprocess.PIPE)
    str_box = p.stdout.readline()
    
    box_conf = ((0,0), (1,1), (2,2),
                (0,1), (0,2), (1,0),
                (1,2), (2,0), (2,1))
    box_tensor = np.zeros((3,3))
    float_box = [float(elem) for elem in str_box.split()]
    for elem, pos in zip(float_box, box_conf):
        box_tensor[pos] = elem
    h    = box_tensor
    hinv = LA.inv(box_tensor)

    # Build the topology
    top_table = pd.read_table(fname_top, delim_whitespace=True, 
            keep_default_na = None, usecols=(0, 2, 3, 4, 6,), 
            header=None, names=('atmid', 'resid', 'mol','atm','q'))

    # Locate the CDC molecules in the topology table
    cdc_atomind = top_table.atm[top_table.mol==args.cdc_molname].index.tolist()
    cdc_numatoms = len(cdc_atomind)
    cdc_numatompmol = cdc_numatoms / args.cdc_molnum
    if (cdc_numatoms % args.cdc_molnum != 0):
        raise ValueError("cdc_molnum ({}) does not divide into the number "
                "of atoms with molecule type cdc_molname ({}:{})".format(
                args.cdc_molnum, args.cdc_molname, cdc_numatoms))
    # Read the CDC table
    cdc_q = pd.read_table(fname_cdc, delim_whitespace=True, header=None, 
            names=('atm', 'qta', 'q0a', 'q1a', 'qtb', 'q0b', 'q1b'),
            keep_default_na = None)

    # Find index boundaries for all non-solvent regions
    nonsolvent_sel = (top_table.mol != "SOL") & (top_table.mol != "NA")
    nonsolvent = top_table[nonsolvent_sel]
    boundary = np.zeros(len(nonsolvent), dtype=np.bool)
    # Do not denote a boundary at index 0 (even though there sorta is one)
    #   because of the method of populating the env_group_mtx
    boundary[0] = 1
    boundary[1:] += ( nonsolvent.resid[1:] != nonsolvent.resid[:-1] )
    ###########################################################################
    # WARNING::: REQUIRES FEWER THAN 100,000 NON-SOLVENT ATOMS (or luck to 
    #   prevent integer wrap-around artifact in atmid-based boundary detection)
    boundary[1:] += ( nonsolvent.atmid[1:] <= nonsolvent.atmid[:-1] )
    ###########################################################################
    nonsolvent_list = top_table.index[nonsolvent_sel].tolist()
    nonsolvent_list = np.array(nonsolvent_list)
    boundary_list = nonsolvent_list[boundary]
    logging.debug( len(boundary_list) )
    logging.debug( boundary_list )


    def mtx_create(boundary_list, top_table, cdc_numatompmol, len_nonsolvent_sel):
        # Create a matrix to project environment atoms into atom groups
        env_group_mtx = np.zeros( (len(boundary_list), len(top_table) - cdc_numatompmol) )
        ###########################################################################
        # WARNING::: REQUIRES THAT BCL MOLECULE COME JUST BEFORE SOLVENT IN THE 
        #   TOPOLOGY - the env_group_mtx is a map from [0:bcl_m)+(bcl_m:end] and 
        #   the boundary index tracks atoms [0:solvent), and to prevent counting
        #   the bcl_m group as an interacting group, we can exclude the final
        #   bcl boundary group.
        #   In other words, a zero-row is included in env_group_mtx to pad for
        #   bcl_m.
        for boundary_index in xrange(len(boundary_list)-1):
        ###########################################################################
            start = boundary_list[boundary_index]
            if boundary_index + 2 == len(boundary_list):
                ###################################################################
                # WARNING::: REQUIRES THAT THE NONSOLVENT SECTION COME BEFORE
                #   THE SOLVENT SECTION, AND THAT THE SOLVENT SECTION CONTINUE
                #   UNTIL THE EOF - the end-index (open-set end) for the final 
                #   nonsolvent group has the value of the number of nonsolvent
                #   atoms, and the solvent group continues until the end of the
                #   array
                end = len_nonsolvent_sel
                ###################################################################
            else:
                end = boundary_list[boundary_index + 1]
            env_group_mtx[boundary_index, start:end] = 1
        env_group_mtx = scipy.sparse.csr_matrix(env_group_mtx)
        return env_group_mtx

    env_group_mtx = mtx_create(boundary_list, top_table, cdc_numatompmol, len(nonsolvent_sel))


    

    
    # Get the coordinates for the full system
    i = 1
    N_skipheader = (  i          * (N_framelines+3) ) + 2
    N_skipfooter = ((11 - i - 1) * (N_framelines+3) ) + 1
    arr_coords = np.genfromtxt(fname_trj, delimiter=(8, 7, 5, 8, 8, 8),
            usecols=(3, 4, 5),
            skip_header = 2, skip_footer = 1)


    def compute_u(cdc_m):
        def get_bclatms(cdc_m):
            startbclm = (cdc_m    ) * cdc_numatompmol
            endbclm   = (cdc_m + 1) * cdc_numatompmol
            cdc_m_atomind = cdc_atomind[startbclm:endbclm]
            cdc_bclmol = top_table.loc[cdc_m_atomind]
            cdc_bclmol = cdc_bclmol.reset_index().merge(cdc_q, on='atm', how='inner').set_index('index')
            if len(cdc_bclmol) != len(cdc_q):
                raise ValueError("Not all atoms in CDC file were found in gro file"
                        "(seeking {} found {})".format(len(cdc_q), len(cdc_bclmol)))
            return cdc_m_atomind, cdc_bclmol

        # cdc_m_atomind: absolute atomic index for bcl_m (??)
        # cdc_bclmol   : topology for bcl_m molecule
        cdc_m_atomind, cdc_bclmol = get_bclatms(cdc_m)
       
         
        #-------------------- --------------------
        # Because arr_coords is 0-based, cdc_bclmol.index is 1-based,
        #   we must shift by one.
        # atm_bcl  : coordinates for bcl molecule, looked up in arr_coords
        # dq_bcl   : dq for bcl, looked up in cdc_bclmol
        atm_bcl = arr_coords[cdc_bclmol.index - 1]
        dq_bcl  = cdc_bclmol.q1a - cdc_bclmol.q0a
        bcl_m_first_ind = min(cdc_m_atomind)
        bcl_m_last_ind  = max(cdc_m_atomind)

        #-------------------- --------------------
        # Numpy magic obfustatingly handles all of the 0-based index problems
        def get_env():
            atm_env_beg = arr_coords[:(bcl_m_first_ind-1), :]
            atm_env_end = arr_coords[bcl_m_last_ind:, :]
            q_env_beg   = top_table.q.values[:(bcl_m_first_ind-1)]
            q_env_end   = top_table.q.values[bcl_m_last_ind:]     
            atm_env = np.concatenate( (atm_env_beg, atm_env_end) )
            q_env   = np.concatenate( (q_env_beg,   q_env_end)   )
            return atm_env, q_env
        atm_env, q_env = get_env()
       
        # Compute with periodic boundary conditions 
        def compute_u_pbc():
            s_bcl = np.tensordot(atm_bcl, hinv, axes=(1, 1))
            s_env = np.tensordot(atm_env, hinv, axes=(1, 1))
            s_ij  = s_env[:, np.newaxis, :] - s_bcl[np.newaxis, :, :]
            s_ij -= np.rint(s_ij)
            r_ij  = np.tensordot(s_ij, h, axes=(2,1))
            rmag_ij= np.sqrt(np.sum(np.square(r_ij), axis=2))
            oneoverr_ij = np.reciprocal(rmag_ij)
            q_ij = q_env[:, np.newaxis] * dq_bcl[np.newaxis, :]
            K_e2nm_cm1 = 1.16E4
            screening = 1./3.
            U_ij = K_e2nm_cm1 * screening * q_ij * oneoverr_ij
            U_atm = np.sum(U_ij, axis=1)
            return U_atm
        U_atm = compute_u_pbc()
        if args.total:
            print(np.sum(U_atm))
        else:
            U_group = env_group_mtx.dot(U_atm)
            ###################################################################
            # WARNING::: REQUIRES THAT THE NONSOLVENT SECTION COME BEFORE
            #   THE SOLVENT SECTION, AND THAT THE SOLVENT SECTION CONTINUE
            #   UNTIL THE EOF - the end-index (open-set end) for the final 
            #   nonsolvent group has the value of the number of nonsolvent
            #   atoms, and the solvent group continues until the end of the
            #   array
            if args.nosolv:
                padding_offset = cdc_m + 1
            else: 
                padding_offset = cdc_m
            U_bclm = U_group[-1]
            for mi in xrange(7-padding_offset):
                U_group[-(1+mi)] = U_group[-(2+mi)]
            U_group[-(1+(7-padding_offset))] = U_bclm
            ###################################################################
            warnings("Length of U_group: {}".format(len(U_group)))
            if args.res is None:
                print(" ".join([str(U) for U in U_group]))
            else:
                print(str(U_group[args.res-1]))

    for cdc_m in xrange(args.cdc_molnum):
        compute_u(cdc_m)





if __name__=="__main__":
    main()
