import numpy as np
import matplotlib as plt

def main():
    fname_trj="data/FMO-md-short.gro"
    fname_top="data/4BCL_pp.top"
    N_framelines = 0
    str_box      = ""
    with open(fname_trj) as f:
        f.readline()
        N_framelines = int(f.readline().strip())
        for i,l in enumerate(f):
            if i == N_framelines:
                str_box = l
                break
    print N_framelines
    print str_box
    arr_charge = np.loadtxt(fname_top, comments=';', usecols=(6,))
    i = 1
    N_skipheader = (  i          * (N_framelines+3) ) + 2
    N_skipfooter = ((11 - i - 1) * (N_framelines+3) ) + 1
    print "Skipping {} and {}".format(N_skipheader, N_skipfooter)
    arr_coords = np.genfromtxt(fname_trj, delimiter=(8, 7, 5, 8, 8, 8),
            usecols=(3, 4, 5),
            skip_header = N_skipheader, skip_footer = N_skipfooter)
    print arr_coords.shape



if __name__=="__main__":
    main()
