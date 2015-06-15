import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib as plt
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-groframe', required=True)
    parser.add_argument('-qtop', required=True)
    parser.add_argument('-qcdc', required=True)
    parser.add_argument('-cdc_molname', type=str, default='BCL')
    parser.add_argument('-cdc_molnum', type=int, default=7)
    args = parser.parse_args()
    fname_trj=args.groframe
    fname_top=args.qtop
    fname_cdc=args.qcdc

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

    arr_coords = np.genfromtxt(fname_trj, delimiter=(8, 7, 5, 8, 8, 8),
            usecols=(3, 4, 5), skip_header = 2, skip_footer = 1)

    # Build the topology
    top_table = pd.read_table(fname_top, delim_whitespace=True, 
            keep_default_na = None, usecols=(3, 4, 6,), 
            header=None, names=('mol','atm','q'))
    cdc_atomind = top_table.atm[top_table.mol==args.cdc_molname].index.tolist()
    cdc_numatoms = len(cdc_atomind)
    cdc_numatompmol = cdc_numatoms / args.cdc_molnum
    if (cdc_numatoms % args.cdc_molnum != 0):
        raise ValueError("cdc_molnum ({}) does not divide into the number "
                "of atoms with molecule type cdc_molname ({}:{})".format(
                args.cdc_molnum, args.cdc_molname, cdc_numatoms))
    
    # Get the coordinates for the full system
    i = 1
    N_skipheader = (  i          * (N_framelines+3) ) + 2
    N_skipfooter = ((11 - i - 1) * (N_framelines+3) ) + 1
    arr_coords = np.genfromtxt(fname_trj, delimiter=(8, 7, 5, 8, 8, 8),
            usecols=(3, 4, 5),
            skip_header = 2, skip_footer = 1)
    
    cdc_q = pd.read_table(fname_cdc, delim_whitespace=True, header=None, 
            names=('atm', 'qta', 'q0a', 'q1a', 'qtb', 'q0b', 'q1b'),
            keep_default_na = None)

    U_m = [] 
    for cdc_m in xrange(args.cdc_molnum):
        start = (cdc_m    ) * cdc_numatompmol
        end   = (cdc_m + 1) * cdc_numatompmol
        cdc_m_atomind = cdc_atomind[start:end]
        cdc_bclmol = top_table.loc[cdc_m_atomind]
        # cdc_bclmol = pd.merge(cdc_bclmol, cdc_q, how='left', on='atm')
        cdc_bclmol = cdc_bclmol.reset_index().merge(cdc_q, on='atm', how='inner').set_index('index')
        if len(cdc_bclmol) != len(cdc_q):
            raise ValueError("Not all atoms in CDC file were found in gro file"
                    "(seeking {} found {})".format(len(cdc_q), len(cdc_bclmol)))

        #-------------------- --------------------
        # NEED TO SUBTRACT 1 FOR 0-BASED INDEX
        atm_bcl = arr_coords[cdc_bclmol.index - 1]
        dq_bcl  = cdc_bclmol.q1a - cdc_bclmol.q0a
        #-------------------- --------------------
        # Numpy magic obfustatingly handles all of the 0-based index problems
        first_ind = min(cdc_m_atomind)
        last_ind  = max(cdc_m_atomind)
        atm_env_beg = arr_coords[:(first_ind-1), :]
        atm_env_end = arr_coords[last_ind:, :]
        q_env_beg   = top_table.q.values[:(first_ind-1)]
        q_env_end   = top_table.q.values[last_ind:]     
        atm_env = np.concatenate( (atm_env_beg, atm_env_end) )
        q_env   = np.concatenate( (q_env_beg,   q_env_end)   )
        
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
        U_tot = np.sum(U_ij, axis=(0, 1))
        U_m.append("{}".format(U_tot))
    print ", ".join(U_m)




if __name__=="__main__":
    main()
