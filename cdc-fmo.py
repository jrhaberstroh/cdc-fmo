from __future__ import print_function
import numpy as np
import numpy.linalg as LA
import numpy.random as RA
import scipy.sparse
import pandas as pd
import matplotlib as plt
import subprocess
import argparse
import logging
import sys

env_atm = 4025 # Environment atom index to debug with

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
    parser.add_argument('-proto', action='store_true')
    parser.add_argument('-indie', action='store_true')
    parser.add_argument('-noise', type=float, default=None, help="Amount of noise to add to position variables")
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

    # [3] M. E. Tuckerman. Statistical Mechanics: Theory and Molecular Simulation. Oxford University Press, Oxford, UK, 2010.
    # 
    # Tuckerman [3] in Appendix B, Eq B.9 provides the general algorithm for the minimum image convention:
    # 
    #   h := [a, b, c], a=(a1,a2,a3), ... (the matrix of box vectors)
    #   r_ij := r_i - r_j                 (difference vector)
    # 
    #   s_i = h^{-1} r_i  
    #   s_ij = s_i - s_j
    #   s_ij <-- s_ij - NINT(s_ij)        (general minimum image convention)
    #   r_ij = h s_ij
    # 
    # NINT(x) is the nearest integer function (e.g. numpy.rint()), h^{-1} is the matrix-inverse of h.
    #
    # Then, from http://manual.gromacs.org/current/online/gro.html
    # * box vectors (free format, space separated reals), values: v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
    # 
    # I interpret this to mean that rows of H are v1, v2, and v3, and the columns are (x), (y), and (z).
    # This means that we have 0,0; 1,1; 2,2; 0,1; 0,2; 1,0;, 1,2; 2,0; 2,1;
    # 
    # AFTERWARDS:
    #  If |r_ij| > 1/4 min( box_xx, box_yy, box_zz )...
    #    * Search all neighboring displacements
    #    * Select the displacement with the smallest |r_ij|
    # 

    #  void pbc_dx(const t_pbc *pbc, const rvec x1, const rvec x2, rvec dx)
    #  {
    #      int      i, j;
    #      rvec     dx_start, trial;
    #      real     d2min, d2trial;
    #      gmx_bool bRot;
    #  
    #      rvec_sub(x1, x2, dx);
    #  
    #      switch (pbc->ePBCDX)
    #      {
    #          case epbcdxRECTANGULAR:
    #              for (i = 0; i < DIM; i++)
    #              {
    #                  while (dx[i] > pbc->hbox_diag[i])
    #                  {
    #                      dx[i] -= pbc->fbox_diag[i];
    #                  }
    #                  while (dx[i] <= pbc->mhbox_diag[i])
    #                  {
    #                      dx[i] += pbc->fbox_diag[i];
    #                  }
    #              }
    #              break;
    #          case epbcdxTRICLINIC:
    #              for (i = DIM-1; i >= 0; i--)
    #              {
    #                  while (dx[i] > pbc->hbox_diag[i])
    #                  {
    #                      for (j = i; j >= 0; j--)
    #                      {
    #                          dx[j] -= pbc->box[i][j];
    #                      }
    #                  }
    #                  while (dx[i] <= pbc->mhbox_diag[i])
    #                  {
    #                      for (j = i; j >= 0; j--)
    #                      {
    #                          dx[j] += pbc->box[i][j];
    #                      }
    #                  }
    #              }
    #              /* dx is the distance in a rectangular box */
    #              d2min = norm2(dx);
    #              if (d2min > pbc->max_cutoff2)
    #              {
    #                  copy_rvec(dx, dx_start);
    #                  d2min = norm2(dx);
    #                  /* Now try all possible shifts, when the distance is within max_cutoff
    #                   * it must be the shortest possible distance.
    #                   */
    #                  i = 0;
    #                  while ((d2min > pbc->max_cutoff2) && (i < pbc->ntric_vec))
    #                  {
    #                      rvec_add(dx_start, pbc->tric_vec[i], trial);
    #                      d2trial = norm2(trial);
    #                      if (d2trial < d2min)
    #                      {
    #                          copy_rvec(trial, dx);
    #                          d2min = d2trial;
    #                      }
    #                      i++;
    #                  }
    #              }
    #              break;
    #      }
    #  }
    # 
    # real max_cutoff2(int ePBC, matrix box)
    # {
    #     real min_hv2, min_ss;
    # 
    #     /* Physical limitation of the cut-off
    #      * by half the length of the shortest box vector.
    #      */
    #     min_hv2 = min(0.25*norm2(box[XX]), 0.25*norm2(box[YY]));
    #     if (ePBC != epbcXY)
    #     {
    #         min_hv2 = min(min_hv2, 0.25*norm2(box[ZZ]));
    #     }
    # 
    #     /* Limitation to the smallest diagonal element due to optimizations:
    #      * checking only linear combinations of single box-vectors (2 in x)
    #      * in the grid search and pbc_dx is a lot faster
    #      * than checking all possible combinations.
    #      */
    #     if (ePBC == epbcXY)
    #     {
    #         min_ss = min(box[XX][XX], box[YY][YY]);
    #     }
    #     else
    #     {
    #         min_ss = min(box[XX][XX], min(box[YY][YY]-fabs(box[ZZ][YY]), box[ZZ][ZZ]));
    #     }
    # 
    #     return min(min_hv2, min_ss*min_ss);
    # }

    box_conf = ((0,0), (1,1), (2,2),
                (0,1), (0,2), (1,0),
                (1,2), (2,0), (2,1))
    box_tensor = np.zeros((3,3))
    float_box = [float(elem) for elem in str_box.split()]
    for elem, pos in zip(float_box, box_conf):
        box_tensor[pos] = elem
    h    = box_tensor.T
    logging.debug(h)
    hinv = LA.inv(h)
    max_cutoff2 = .25 * np.amin(np.diag(h))
    assert(np.tensordot(hinv, h, (1,0))[0,0] - 1.0 < 0.00001)

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

    if not args.noise is None: 
        arr_coords += RA.uniform(-args.noise/2.0, args.noise/2.0, arr_coords.shape)

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
        # 2015-11-??: Because arr_coords is 0-based, cdc_bclmol.index is 1-based,
        #             we must shift by one.
        # 2016-02-10: In A/B comparison test to the much less complicated 
        #             umbrellamacs code, it suggests that we should exclude the
        #             - 1 index shift correction. 
        #       TODO: Resolve WHY this shift is unnecessary, or if umbrellamacs
        #             is the real source of the problem.
        # atm_bcl - coordinates for bcl molecule, looked up in arr_coords
        # dq_bcl  - dq for bcl, looked up in cdc_bclmol
        atm_bcl = arr_coords[cdc_bclmol.index]#  - 1]
        dq_bcl  = cdc_bclmol.q1b - cdc_bclmol.q0b
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
            s_ijx  = s_env[:, np.newaxis, :] - s_bcl[np.newaxis, :, :]
            s_ijx -= np.rint(s_ijx)
            r_ijx  = np.tensordot(s_ijx, h, axes=(2,1))
            rmag_ij= np.sqrt(np.sum(np.square(r_ijx), axis=2))

            # Check all adjacent neighbors if the initial distance is greater
            # than max_cutoff2 to use the compact representation
            toolong = rmag_ij > max_cutoff2
            rstart_kx = np.copy(r_ijx[toolong])
            rfix_kx   = np.copy(r_ijx[toolong])
            rmagfix_k = np.copy(rmag_ij[toolong])
            r_ijx_changed = False
            for i0 in xrange(12):
                pm = ((i0 % 2) * 2) - 1
                i  = i0 / 2
                if i in [0, 1, 2]:
                    latvec = h[np.newaxis, :, i]
                    rshift_kx = rstart_kx + pm * latvec
                elif i in [3, 4, 5]:
                    i -= 3
                    if i < 2:
                        latvec = h[np.newaxis, :, 2] - h[np.newaxis, :, i]
                    elif i == 2:
                        latvec = h[np.newaxis, :, 2] - h[np.newaxis, :, 0] - h[np.newaxis, :, 1] 
                    rshift_kx = rstart_kx + pm * latvec

                rshiftmag_k = np.sqrt(np.sum(np.square(rshift_kx), axis=1))
                replace_ind = rshiftmag_k < rmagfix_k
                if np.any(replace_ind):
                    r_ijx_changed=True
                    logging.debug("Replacing {} values".format(np.sum(replace_ind)))
                    # logging.debug(rstart_kx.shape)
                    # logging.debug(rshift_kx.shape)
                    # logging.debug(rshift_kx[replace_ind].shape)
                    # logging.debug(r_ijx[toolong].shape)
                    # logging.debug(r_ijx[toolong][replace_ind].shape)
                    rfix_kx[replace_ind]   = rshift_kx[replace_ind, :]
                    rmagfix_k[replace_ind] = rshiftmag_k[replace_ind]
            r_ijx[toolong] = rfix_kx
            rmag_ij[toolong] = rmagfix_k

            oneoverr_ij = np.reciprocal(rmag_ij)
            q_ij = q_env[:, np.newaxis] * dq_bcl[np.newaxis, :]
            K_e2nm_cm1 = 1.16E4
            screening = .3333
            # i and j are non-intersecting, so no need to avoid self-interaction
            U_ij = K_e2nm_cm1 * screening * q_ij * oneoverr_ij
            if cdc_m == 1:
                logging.debug("BCL %d: %f %f %f (%f)", cdc_m + 367, atm_bcl[0,0], atm_bcl[0,1], atm_bcl[0,2], dq_bcl.values[0])
                logging.debug("ENV %d: %f %f %f (%f)", env_atm+1, atm_env[env_atm,0], atm_env[env_atm,1], atm_env[env_atm,2], q_env[env_atm])
                logging.debug("DIST %d: %f", cdc_m + 367, rmag_ij[env_atm,0])
                logging.debug("KES %d: %f", cdc_m + 367, K_e2nm_cm1 / 349.757)
                logging.debug("INTQ %d: %f", cdc_m + 367, q_ij[env_atm,0])
                logging.debug("INTR %d: %f", cdc_m + 367, U_ij[env_atm,0])
            U_atm = np.sum(U_ij, axis=1)
            warnings("U_ij shape: {}".format(U_ij.shape))
            return U_atm
        U_atm = compute_u_pbc()
        if args.total:
            print(np.sum(U_atm))
        elif args.indie:
            print(" ".join([str(U) for U in U_atm]))
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

    def proto_compute_u(cdc_m):
        BCL4_resnr = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,
        9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,
        10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,
        11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,
        12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,
        13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,
        14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,
        16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,
        17,17,17,17,17,17,17,17,17,17,17,18,18,18,18,
        18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,
        19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,
        19,19,19,19,20,20,20,20,20,20,20,20,20,20,20,
        20,20,20,20,21,21,21,21,21,21,21,22,22,22,22,
        22,22,22,23,23,23,23,23,23,23,23,23,23,23,24,
        24,24,24,24,24,24,24,24,24,24,25,25,25,25,25,
        25,25,25,25,25,25,26,26,26,26,26,26,26,26,26,
        26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,
        27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,
        28,28,28,28,28,28,28,28,28,29,29,29,29,29,29,
        29,29,29,29,29,29,29,29,29,29,30,30,30,30,30,
        30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
        30,30,31,31,31,31,31,31,31,32,32,32,32,32,32,
        32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
        32,32,32,33,33,33,33,33,33,33,33,33,33,34,34,
        34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,
        34,34,34,34,34,35,35,35,35,35,35,35,35,35,35,
        35,35,35,35,35,35,36,36,36,36,36,36,36,36,36,
        36,36,36,36,36,37,37,37,37,37,37,37,37,37,37,
        37,37,37,37,37,37,38,38,38,38,38,38,38,38,38,
        38,38,38,38,38,39,39,39,39,39,39,39,39,39,39,
        40,40,40,40,40,40,40,40,40,40,41,41,41,41,41,
        41,41,41,41,41,41,41,41,41,41,41,41,41,41,42,
        42,42,42,42,42,42,42,42,42,42,42,42,42,43,43,
        43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,
        43,43,44,44,44,44,44,44,44,44,44,44,44,44,44,
        44,44,44,44,44,44,45,45,45,45,45,45,45,45,45,
        45,45,45,45,45,46,46,46,46,46,46,46,46,46,46,
        46,46,46,46,47,47,47,47,47,47,47,47,47,47,47,
        47,48,48,48,48,48,48,48,48,48,48,48,49,49,49,
        49,49,49,49,49,49,49,49,49,49,49,50,50,50,50,
        50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,
        51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,
        51,51,51,51,51,51,51,51,51,52,52,52,52,52,52,
        52,52,52,52,52,52,52,52,52,52,52,52,52,53,53,
        53,53,53,53,53,53,53,53,53,53,54,54,54,54,54,
        54,54,54,54,54,55,55,55,55,55,55,55,55,55,55,
        55,55,55,55,55,55,55,55,55,55,55,55,56,56,56,
        56,56,56,56,56,56,56,56,56,56,56,57,57,57,57,
        57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,
        58,58,58,58,58,58,58,58,58,58,58,58,59,59,59,
        59,59,59,59,60,60,60,60,60,60,60,60,60,60,60,
        60,60,60,60,60,61,61,61,61,61,61,61,61,61,61,
        61,61,61,61,61,61,62,62,62,62,62,62,62,62,62,
        62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,
        63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,
        63,63,63,63,63,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,65,65,65,65,65,65,65,65,65,65,65,
        65,65,65,66,66,66,66,66,66,66,66,66,66,66,66,
        66,66,66,66,66,66,66,66,66,66,67,67,67,67,67,
        67,67,67,67,67,67,67,67,67,67,67,67,67,67,68,
        68,68,68,68,68,68,68,68,68,68,68,68,68,68,69,
        69,69,69,69,69,69,69,69,69,69,70,70,70,70,70,
        70,70,70,70,70,70,70,70,70,70,70,71,71,71,71,
        71,71,71,71,71,71,71,71,71,71,71,71,72,72,72,
        72,72,72,72,72,72,72,72,72,73,73,73,73,73,73,
        73,73,73,73,73,74,74,74,74,74,74,74,74,74,74,
        74,74,74,74,74,74,75,75,75,75,75,75,75,75,75,
        75,75,75,75,75,75,75,75,75,75,75,75,75,76,76,
        76,76,76,76,76,76,76,76,76,76,76,76,77,77,77,
        77,77,77,77,77,77,77,77,77,77,77,78,78,78,78,
        78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,
        79,79,79,79,79,79,79,79,79,79,79,79,79,79,80,
        80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,
        81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,
        82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,
        82,83,83,83,83,83,83,83,83,83,83,83,83,84,84,
        84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,
        84,84,85,85,85,85,85,85,85,85,85,85,86,86,86,
        86,86,86,86,86,86,86,86,86,86,86,87,87,87,87,
        87,87,87,87,87,87,87,87,87,87,87,88,88,88,88,
        88,88,88,88,88,88,88,88,88,88,89,89,89,89,89,
        89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,
        89,89,90,90,90,90,90,90,90,90,90,90,90,90,91,
        91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,
        91,91,91,91,91,91,91,91,92,92,92,92,92,92,92,
        92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,
        92,92,93,93,93,93,93,93,93,93,93,93,93,93,93,
        93,93,93,93,93,93,94,94,94,94,94,94,94,94,94,
        94,95,95,95,95,95,95,95,95,95,95,95,95,95,95,
        95,95,96,96,96,96,96,96,96,97,97,97,97,97,97,
        97,97,97,97,97,97,97,97,97,98,98,98,98,98,98,
        98,99,99,99,99,99,99,99,99,99,99,99,100,100,100,
        100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
        100,101,101,101,101,101,101,101,101,101,101,101,102,102,102,
        102,102,102,102,102,102,102,102,102,102,102,102,102,103,103,
        103,103,103,103,103,104,104,104,104,104,104,104,104,104,104,
        104,104,105,105,105,105,105,105,105,105,105,105,105,105,105,
        105,105,105,105,105,105,105,106,106,106,106,106,106,106,106,
        106,106,106,107,107,107,107,107,107,107,107,107,107,107,107,
        107,107,107,107,107,108,108,108,108,108,108,108,108,108,108,
        108,109,109,109,109,109,109,109,109,109,109,109,109,109,109,
        109,109,109,109,109,109,110,110,110,110,110,110,110,110,110,
        110,110,111,111,111,111,111,111,111,111,111,111,111,111,111,
        111,111,111,111,111,111,111,112,112,112,112,112,112,112,112,
        112,112,112,112,112,112,112,113,113,113,113,113,113,113,114,
        114,114,114,114,114,114,114,114,114,114,115,115,115,115,115,
        115,115,115,115,115,115,115,115,115,115,115,116,116,116,116,
        116,116,116,116,116,116,116,116,116,116,116,116,117,117,117,
        117,117,117,117,117,117,117,117,117,117,117,118,118,118,118,
        118,118,118,118,118,118,118,118,118,118,118,118,118,119,119,
        119,119,119,119,119,119,119,119,119,119,119,119,119,119,119,
        119,119,119,119,120,120,120,120,120,120,120,120,120,120,120,
        120,120,120,120,120,120,120,120,120,120,121,121,121,121,121,
        121,121,121,121,121,121,121,121,121,121,121,121,121,121,121,
        121,122,122,122,122,122,122,122,122,122,122,122,122,122,122,
        122,122,122,122,122,122,122,122,122,122,123,123,123,123,123,
        123,123,123,123,123,123,124,124,124,124,124,124,124,124,124,
        124,124,124,125,125,125,125,125,125,125,125,125,125,126,126,
        126,126,126,126,126,126,126,126,126,126,126,126,126,126,127,
        127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,
        127,127,127,127,127,127,127,127,128,128,128,128,128,128,128,
        128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
        128,128,129,129,129,129,129,129,129,129,129,129,129,129,129,
        129,130,130,130,130,130,130,130,130,130,130,130,130,130,130,
        130,130,130,130,130,131,131,131,131,131,131,131,131,131,131,
        131,131,131,131,132,132,132,132,132,132,132,132,132,132,132,
        132,132,132,133,133,133,133,133,133,133,133,133,133,133,133,
        133,133,134,134,134,134,134,134,134,134,134,134,134,134,134,
        134,134,134,134,134,134,135,135,135,135,135,135,135,135,135,
        135,135,135,135,135,135,135,135,135,135,135,135,136,136,136,
        136,136,136,136,136,136,136,136,136,136,136,136,136,136,137,
        137,137,137,137,137,137,137,137,137,137,137,137,137,137,137,
        137,138,138,138,138,138,138,138,139,139,139,139,139,139,139,
        139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,
        139,139,140,140,140,140,140,140,140,140,140,140,140,140,140,
        140,140,140,140,141,141,141,141,141,141,141,141,141,141,141,
        141,141,141,141,141,141,141,141,141,142,142,142,142,142,142,
        142,142,142,142,142,142,142,142,142,142,142,143,143,143,143,
        143,143,143,143,143,143,143,143,144,144,144,144,144,144,144,
        144,144,144,144,144,144,144,144,144,144,144,144,145,145,145,
        145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,
        145,146,146,146,146,146,146,146,146,146,146,146,146,146,146,
        146,146,146,147,147,147,147,147,147,147,147,147,147,147,147,
        147,147,147,147,147,147,147,147,147,147,148,148,148,148,148,
        148,148,148,148,148,148,148,148,148,148,148,149,149,149,149,
        149,149,149,149,149,149,149,149,149,149,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,150,150,150,150,150,151,
        151,151,151,151,151,151,151,151,151,151,151,152,152,152,152,
        152,152,152,152,152,152,152,152,152,152,153,153,153,153,153,
        153,153,153,153,153,153,153,153,153,154,154,154,154,154,154,
        154,154,154,154,154,154,155,155,155,155,155,155,155,155,155,
        155,155,155,155,155,155,155,155,155,155,156,156,156,156,156,
        156,156,156,156,156,156,156,156,156,156,156,157,157,157,157,
        157,157,157,157,157,157,157,157,158,158,158,158,158,158,158,
        158,158,158,158,158,158,158,159,159,159,159,159,159,159,159,
        159,159,159,159,159,159,159,159,159,159,159,159,159,159,159,
        159,160,160,160,160,160,160,160,160,160,160,160,160,160,160,
        160,161,161,161,161,161,161,161,162,162,162,162,162,162,162,
        162,162,162,162,162,162,162,162,162,162,162,162,162,163,163,
        163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,
        164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,
        164,164,165,165,165,165,165,165,165,165,165,165,165,166,166,
        166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,
        166,166,167,167,167,167,167,167,167,168,168,168,168,168,168,
        168,168,168,168,169,169,169,169,169,169,169,169,169,169,169,
        169,169,169,170,170,170,170,170,170,170,170,170,170,170,170,
        170,170,170,170,170,170,170,170,171,171,171,171,171,171,171,
        172,172,172,172,172,172,172,172,172,172,172,172,173,173,173,
        173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,
        173,173,173,173,173,173,174,174,174,174,174,174,174,174,174,
        174,174,174,174,174,174,174,174,174,174,175,175,175,175,175,
        175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,
        175,175,175,175,176,176,176,176,176,176,176,176,176,176,176,
        176,176,176,176,177,177,177,177,177,177,177,177,177,177,177,
        177,177,177,177,177,177,177,177,177,178,178,178,178,178,178,
        178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,
        178,178,178,179,179,179,179,179,179,179,179,179,179,179,179,
        179,179,179,179,179,179,179,179,180,180,180,180,180,180,180,
        180,180,180,180,180,180,180,180,180,180,180,180,181,181,181,
        181,181,181,181,182,182,182,182,182,182,182,182,182,182,182,
        182,182,182,183,183,183,183,183,183,183,183,183,183,184,184,
        184,184,184,184,184,184,184,184,184,184,184,184,184,184,184,
        184,184,184,185,185,185,185,185,185,185,185,185,185,186,186,
        186,186,186,186,186,186,186,186,187,187,187,187,187,187,187,
        187,187,187,187,187,187,187,187,187,187,187,187,188,188,188,
        188,188,188,188,188,188,188,188,188,188,188,189,189,189,189,
        189,189,189,189,189,189,189,189,189,189,189,190,190,190,190,
        190,190,190,191,191,191,191,191,191,191,192,192,192,192,192,
        192,192,192,192,192,192,192,192,192,192,192,192,193,193,193,
        193,193,193,193,193,193,193,193,193,193,193,193,193,193,193,
        193,193,193,193,193,193,194,194,194,194,194,194,194,194,194,
        194,194,194,194,194,194,194,194,194,194,195,195,195,195,195,
        195,195,195,195,195,195,196,196,196,196,196,196,196,196,196,
        196,196,196,196,196,197,197,197,197,197,197,197,197,197,197,
        197,197,197,197,197,197,197,197,197,198,198,198,198,198,198,
        198,198,198,198,198,198,198,198,198,198,199,199,199,199,199,
        199,199,199,199,199,199,199,199,199,199,199,200,200,200,200,
        200,200,200,200,200,200,200,200,200,200,201,201,201,201,201,
        201,201,201,201,201,201,202,202,202,202,202,202,202,202,202,
        202,202,203,203,203,203,203,203,203,203,203,203,203,203,203,
        203,204,204,204,204,204,204,204,204,204,204,204,204,204,204,
        204,204,205,205,205,205,205,205,205,205,205,205,205,205,205,
        205,205,206,206,206,206,206,206,206,207,207,207,207,207,207,
        207,208,208,208,208,208,208,208,208,208,208,208,208,208,208,
        209,209,209,209,209,209,209,209,209,209,209,209,209,209,209,
        209,210,210,210,210,210,210,210,211,211,211,211,211,211,211,
        211,211,211,211,211,211,211,211,211,212,212,212,212,212,212,
        212,212,212,212,212,212,212,212,213,213,213,213,213,213,213,
        213,213,213,213,213,213,213,213,213,213,213,213,213,213,213,
        213,213,214,214,214,214,214,214,214,214,214,214,214,214,214,
        214,214,214,214,214,214,214,214,214,214,214,215,215,215,215,
        215,215,215,215,215,215,215,215,215,215,215,215,215,215,215,
        215,215,215,216,216,216,216,216,216,216,216,216,216,216,216,
        216,216,216,216,216,216,216,216,217,217,217,217,217,217,217,
        217,217,217,217,218,218,218,218,218,218,218,218,218,218,218,
        218,218,218,218,218,218,219,219,219,219,219,219,219,219,219,
        219,220,220,220,220,220,220,220,221,221,221,221,221,221,221,
        221,221,221,221,222,222,222,222,222,222,222,223,223,223,223,
        223,223,223,223,223,223,223,223,223,223,223,223,224,224,224,
        224,224,224,224,224,224,224,224,224,224,224,224,224,225,225,
        225,225,225,225,225,225,225,225,225,225,226,226,226,226,226,
        226,226,226,226,226,226,227,227,227,227,227,227,227,227,227,
        227,227,227,227,227,227,227,227,227,227,228,228,228,228,228,
        228,228,228,228,228,228,229,229,229,229,229,229,229,229,229,
        229,229,229,229,229,229,229,229,229,229,229,229,229,229,229,
        230,230,230,230,230,230,230,230,230,230,230,230,230,230,230,
        230,230,230,230,230,230,230,230,230,231,231,231,231,231,231,
        231,231,231,231,231,231,231,231,232,232,232,232,232,232,232,
        232,232,232,232,232,232,232,232,233,233,233,233,233,233,233,
        233,233,233,233,233,233,233,233,233,233,233,233,234,234,234,
        234,234,234,234,234,234,234,234,234,234,234,234,234,234,234,
        234,234,235,235,235,235,235,235,235,235,235,235,235,235,235,
        235,236,236,236,236,236,236,236,236,236,236,236,236,236,236,
        236,236,237,237,237,237,237,237,237,237,237,237,237,237,237,
        237,237,238,238,238,238,238,238,238,238,238,238,238,238,238,
        238,238,238,238,239,239,239,239,239,239,239,239,239,239,239,
        239,239,239,239,239,239,239,239,240,240,240,240,240,240,240,
        240,240,240,240,240,240,240,241,241,241,241,241,241,241,241,
        241,241,241,241,241,241,241,241,241,241,241,241,241,241,242,
        242,242,242,242,242,242,242,242,242,242,242,242,242,243,243,
        243,243,243,243,243,243,243,243,244,244,244,244,244,244,244,
        244,244,244,244,245,245,245,245,245,245,245,245,245,245,245,
        245,245,245,245,245,245,245,245,246,246,246,246,246,246,246,
        246,246,246,246,246,246,246,246,247,247,247,247,247,247,247,
        248,248,248,248,248,248,248,249,249,249,249,249,249,249,249,
        249,249,249,249,249,249,249,249,249,249,249,249,250,250,250,
        250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,
        250,250,250,250,250,250,251,251,251,251,251,251,251,251,251,
        251,251,252,252,252,252,252,252,252,252,252,252,252,252,253,
        253,253,253,253,253,253,253,253,253,253,254,254,254,254,254,
        254,254,254,254,254,254,254,254,254,254,254,254,255,255,255,
        255,255,255,255,256,256,256,256,256,256,256,256,256,256,256,
        256,256,256,256,256,256,256,256,257,257,257,257,257,257,257,
        257,257,257,257,257,257,257,257,258,258,258,258,258,258,258,
        258,258,258,258,258,258,258,258,258,259,259,259,259,259,259,
        259,259,259,259,259,259,259,259,259,259,259,259,259,259,259,
        259,260,260,260,260,260,260,260,260,260,260,260,260,260,260,
        260,260,261,261,261,261,261,261,261,261,261,261,261,261,262,
        262,262,262,262,262,262,263,263,263,263,263,263,263,263,263,
        263,263,263,263,263,264,264,264,264,264,264,264,264,264,264,
        264,264,264,264,264,264,264,264,264,265,265,265,265,265,265,
        265,265,265,265,265,265,265,265,266,266,266,266,266,266,266,
        267,267,267,267,267,267,267,267,267,267,267,267,267,267,267,
        267,268,268,268,268,268,268,268,268,268,268,268,269,269,269,
        269,269,269,269,269,269,269,269,269,269,269,269,269,269,269,
        269,269,269,269,269,269,270,270,270,270,270,270,270,270,270,
        270,270,270,271,271,271,271,271,271,271,271,271,271,272,272,
        272,272,272,272,272,273,273,273,273,273,273,273,274,274,274,
        274,274,274,274,275,275,275,275,275,275,275,275,275,275,275,
        275,275,275,275,275,275,275,275,276,276,276,276,276,276,276,
        276,276,276,276,276,276,276,276,276,276,276,276,276,276,276,
        276,276,277,277,277,277,277,277,277,277,277,277,277,277,277,
        277,277,277,277,277,277,277,277,277,277,277,278,278,278,278,
        278,278,278,278,278,278,278,278,278,278,278,278,278,278,278,
        279,279,279,279,279,279,279,279,279,279,279,279,279,279,279,
        279,279,279,279,280,280,280,280,280,280,280,280,280,280,280,
        280,280,280,281,281,281,281,281,281,281,281,281,281,281,281,
        281,281,281,281,281,282,282,282,282,282,282,282,282,282,282,
        282,282,282,282,283,283,283,283,283,283,283,283,283,283,283,
        283,283,283,283,283,283,283,283,284,284,284,284,284,284,284,
        284,284,284,284,284,284,284,284,284,284,284,284,285,285,285,
        285,285,285,285,285,285,285,285,285,285,285,286,286,286,286,
        286,286,286,286,286,286,286,286,286,286,286,286,286,286,286,
        287,287,287,287,287,287,287,287,287,287,287,287,287,287,287,
        287,288,288,288,288,288,288,288,288,288,288,288,288,288,288,
        288,288,288,289,289,289,289,289,289,289,289,289,289,289,289,
        289,289,289,289,289,290,290,290,290,290,290,290,291,291,291,
        291,291,291,291,291,291,291,291,291,291,291,291,291,291,292,
        292,292,292,292,292,292,292,292,292,292,292,292,292,292,292,
        293,293,293,293,293,293,293,294,294,294,294,294,294,294,294,
        294,294,294,294,294,294,294,294,294,294,294,294,294,294,295,
        295,295,295,295,295,295,295,295,295,295,295,295,295,295,295,
        295,295,295,295,296,296,296,296,296,296,296,296,296,296,296,
        296,296,296,297,297,297,297,297,297,297,297,297,297,297,297,
        298,298,298,298,298,298,298,298,298,298,298,298,298,298,298,
        298,298,298,298,298,299,299,299,299,299,299,299,299,299,299,
        299,299,299,299,300,300,300,300,300,300,300,300,300,300,300,
        300,300,300,300,300,301,301,301,301,301,301,301,301,301,301,
        301,301,302,302,302,302,302,302,302,302,302,302,302,302,302,
        302,303,303,303,303,303,303,303,303,303,303,303,303,303,303,
        303,303,303,304,304,304,304,304,304,304,304,304,304,304,304,
        304,304,304,304,304,304,304,305,305,305,305,305,305,305,305,
        305,305,305,305,305,305,305,305,305,305,305,305,305,305,306,
        306,306,306,306,306,306,306,306,306,306,306,306,306,306,306,
        306,306,306,307,307,307,307,307,307,307,307,307,307,307,307,
        307,307,307,307,308,308,308,308,308,308,308,308,308,308,308,
        308,308,308,308,308,308,308,308,309,309,309,309,309,309,309,
        309,309,309,309,309,309,309,310,310,310,310,310,310,310,310,
        310,310,310,310,310,310,310,310,310,310,310,310,310,310,311,
        311,311,311,311,311,311,312,312,312,312,312,312,312,312,312,
        312,312,312,312,312,312,312,312,312,312,312,312,313,313,313,
        313,313,313,313,313,313,313,313,313,313,313,313,313,313,313,
        313,313,313,313,314,314,314,314,314,314,314,314,314,314,314,
        314,314,314,314,314,314,314,314,315,315,315,315,315,315,315,
        315,315,315,315,315,315,315,315,315,315,315,315,315,315,315,
        315,315,316,316,316,316,316,316,316,316,316,316,316,316,316,
        316,316,316,316,316,316,316,316,317,317,317,317,317,317,317,
        317,317,317,318,318,318,318,318,318,318,318,318,318,319,319,
        319,319,319,319,319,319,319,319,319,319,319,319,320,320,320,
        320,320,320,320,320,320,320,320,320,320,320,320,320,320,321,
        321,321,321,321,321,321,321,321,321,321,321,321,321,321,321,
        321,321,321,321,322,322,322,322,322,322,322,322,322,322,322,
        322,322,322,322,322,322,322,322,322,322,322,322,322,323,323,
        323,323,323,323,323,323,323,323,323,324,324,324,324,324,324,
        324,324,324,324,324,324,324,324,324,324,324,325,325,325,325,
        325,325,325,325,325,325,325,325,325,325,326,326,326,326,326,
        326,326,326,326,326,326,326,326,326,326,326,326,326,326,327,
        327,327,327,327,327,327,327,327,327,327,327,327,327,327,328,
        328,328,328,328,328,328,328,328,328,328,328,328,328,328,329,
        329,329,329,329,329,329,329,329,329,329,329,329,329,329,329,
        329,329,329,329,329,330,330,330,330,330,330,330,330,330,330,
        330,330,330,330,330,330,330,330,330,330,330,330,330,330,331,
        331,331,331,331,331,331,331,331,331,331,331,331,331,331,331,
        331,331,331,331,331,331,331,331,332,332,332,332,332,332,332,
        332,332,332,332,333,333,333,333,333,333,333,334,334,334,334,
        334,334,334,335,335,335,335,335,335,335,335,335,335,336,336,
        336,336,336,336,336,336,336,336,336,336,336,336,336,336,336,
        336,336,336,336,337,337,337,337,337,337,337,337,337,337,338,
        338,338,338,338,338,338,338,338,338,338,338,338,338,338,338,
        338,338,338,338,338,338,338,338,339,339,339,339,339,339,339,
        339,339,339,339,339,339,339,339,339,339,339,339,339,339,339,
        339,339,340,340,340,340,340,340,340,340,340,340,340,340,340,
        340,340,340,341,341,341,341,341,341,341,341,341,341,341,341,
        341,341,341,342,342,342,342,342,342,342,342,342,342,342,342,
        342,342,342,342,342,343,343,343,343,343,343,343,343,343,343,
        343,343,343,343,343,343,344,344,344,344,344,344,344,344,344,
        344,344,345,345,345,345,345,345,345,345,345,345,345,345,345,
        345,345,345,345,345,345,345,345,345,346,346,346,346,346,346,
        346,347,347,347,347,347,347,347,348,348,348,348,348,348,348,
        348,348,348,348,348,348,348,349,349,349,349,349,349,349,350,
        350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,
        350,351,351,351,351,351,351,351,351,351,351,351,351,351,351,
        351,351,351,351,351,351,352,352,352,352,352,352,352,352,352,
        352,352,352,352,352,352,353,353,353,353,353,353,353,353,353,
        353,353,353,353,353,353,353,354,354,354,354,354,354,354,354,
        354,354,354,354,354,354,354,354,354,354,354,355,355,355,355,
        355,355,355,355,355,355,355,355,355,355,355,355,355,355,355,
        355,355,356,356,356,356,356,356,356,356,356,356,357,357,357,
        357,357,357,357,357,357,357,357,357,357,357,357,357,357,357])

        bcl_cdc_charges = np.array([0.017,0.027,0.021,0.000,0.053,0.000,0.030,0.000,0.028,-0.020,-0.031,-0.009,0.000,-0.003,0.000,-0.004,0.000,0.000,0.000,-0.004,0.000,0.000,0.001,0.000,0.000,-0.003,0.001,0.001,0.014,-0.023,-0.070,0.027,0.027,0.001,0.000,0.000,0.000,0.023,0.013,0.000,0.000,0.000,0.000,0.039,-0.041,-0.060,-0.005,0.000,-0.003,0.000,-0.003,0.000,0.000,0.000,-0.004,0.000,0.000,-0.001,0.000,0.000,0.000,0.012,-0.053,-0.047,0.018,-0.004,-0.002,0.000,0.000,0.000,0.027,0.009,-0.002,0.000,-0.002,0.002,0.003,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000])

        site_n_couple = np.zeros((max(BCL4_resnr) - min(BCL4_resnr) + 1))

        bcl_cdc_charges = - bcl_cdc_charges
        q_env  = top_table.q.values[0:len(BCL4_resnr)]


        bcl_sta = len(BCL4_resnr) + len(bcl_cdc_charges) * cdc_m
        bcl_fin = len(BCL4_resnr) + len(bcl_cdc_charges) * (cdc_m + 1)
        atm_bcl = arr_coords[bcl_sta:bcl_fin, :]
        atm_env = arr_coords[0:len(BCL4_resnr), :]

        s_bcl = np.tensordot(atm_bcl, hinv, axes=(1, 1))
        s_env = np.tensordot(atm_env, hinv, axes=(1, 1))
        s_ijx  = s_env[:, np.newaxis, :] - s_bcl[np.newaxis, :, :]
        s_ijx -= np.rint(s_ijx)
        r_ijx  = np.tensordot(s_ijx, h, axes=(2,1))
        rmag_ij= np.sqrt(np.sum(np.square(r_ijx), axis=2))

        # Check all adjacent neighbors if the initial distance is greater
        # than max_cutoff2 to use the compact representation
        toolong = rmag_ij > max_cutoff2
        rstart_kx = np.copy(r_ijx[toolong])
        rfix_kx   = np.copy(r_ijx[toolong])
        rmagfix_k = np.copy(rmag_ij[toolong])
        r_ijx_changed = False
        for i0 in xrange(12):
            pm = ((i0 % 2) * 2) - 1
            i  = i0 / 2
            if i in [0, 1, 2]:
                latvec = h[np.newaxis, :, i]
                rshift_kx = rstart_kx + pm * latvec
            elif i in [3, 4, 5]:
                i -= 3
                if i < 2:
                    latvec = h[np.newaxis, :, 2] - h[np.newaxis, :, i]
                elif i == 2:
                    latvec = h[np.newaxis, :, 2] - h[np.newaxis, :, 0] - h[np.newaxis, :, 1] 
                rshift_kx = rstart_kx + pm * latvec

            rshiftmag_k = np.sqrt(np.sum(np.square(rshift_kx), axis=1))
            replace_ind = rshiftmag_k < rmagfix_k
            if np.any(replace_ind):
                r_ijx_changed=True
                logging.debug("Replacing {} values".format(np.sum(replace_ind)))
                # logging.debug(rstart_kx.shape)
                # logging.debug(rshift_kx.shape)
                # logging.debug(rshift_kx[replace_ind].shape)
                # logging.debug(r_ijx[toolong].shape)
                # logging.debug(r_ijx[toolong][replace_ind].shape)
                rfix_kx[replace_ind]   = rshift_kx[replace_ind, :]
                rmagfix_k[replace_ind] = rshiftmag_k[replace_ind]
        r_ijx[toolong] = rfix_kx
        rmag_ij[toolong] = rmagfix_k

        if np.all(r_ijx[toolong]==rstart_kx) and r_ijx_changed:
            raise RuntimeError("r_ijx values not modified despite finding better shifts")

        oneoverr_ij = np.reciprocal(rmag_ij)
        q_ij = q_env[:, np.newaxis] * bcl_cdc_charges[np.newaxis, :]
        K_e2nm_cm1 = 1.16E4
        screening = .3333
        # i and j are non-intersecting, so no need to avoid self-interaction
        U_ij = K_e2nm_cm1 * screening * q_ij * oneoverr_ij
        if cdc_m == 1:
            logging.debug("BCL %d: %f %f %f (%f)", cdc_m + 367, atm_bcl[0,0], atm_bcl[0,1], atm_bcl[0,2], bcl_cdc_charges[0])
            logging.debug("ENV %d: %f %f %f (%f)", cdc_m + 367, atm_env[env_atm,0], atm_env[env_atm,1], atm_env[env_atm,2], q_env[env_atm])
            logging.debug("DIST %d: %f", cdc_m + 367, rmag_ij[env_atm,0])
            logging.debug("KES %d: %f", cdc_m + 367, K_e2nm_cm1 / 349.757)
            logging.debug("INTQ %d: %f", cdc_m + 367, q_ij[env_atm,0])
            logging.debug("INTR %d: %f", cdc_m + 367, U_ij[env_atm,0])
        logging.debug(U_ij.shape)
        U_atom = np.sum(U_ij, axis=1)
        if args.total:
            print(np.sum(U_atom))
        elif args.indie:
            print(" ".join([str(U) for U in U_atom]))
        else:
            for i in xrange(len(BCL4_resnr)):
                site_n_couple[BCL4_resnr[i] - min(BCL4_resnr)] += U_atom[i]
            print(" ".join([str(U) for U in site_n_couple]))
    for cdc_m in xrange(args.cdc_molnum):
        if args.proto:
            logging.info("Running with prototype code")
            proto_compute_u(cdc_m)
        else:
            logging.info("Running with real code")
            compute_u(cdc_m)






if __name__=="__main__":
    main()
