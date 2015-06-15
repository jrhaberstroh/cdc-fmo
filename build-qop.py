from __future__ import print_function
import argparse
import numpy
import sys

def warnings(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

parser=argparse.ArgumentParser()
parser.add_argument('-mol', required=True)
parser.add_argument('-moltype', required=True, nargs='+')
args=parser.parse_args()

moltypes=[]
for fn in args.moltype:
    with open(fn) as f:
        moltypes.append(f.readline().rstrip())

mol_inds=[]
mol_qtys=[]
with open(args.mol) as f:
    for l in f:
        l_arr = l.split()
        mol_ind = moltypes.index(l_arr[0])
        mol_qty = int(l_arr[1].rstrip())
        mol_inds.append(mol_ind)
        mol_qtys.append(mol_qty)

for mol_ind, mol_rep in zip(mol_inds, mol_qtys):
    warnings("Using MOL={}({}) x {}".format(moltypes[mol_ind], mol_ind, mol_rep))
    for rep in xrange(mol_rep):
        with open(args.moltype[mol_ind]) as f:
            # Skip the first line, and print the rest
            f.readline()
            for l in f:
                print(l, end="")

