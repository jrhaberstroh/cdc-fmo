#!/bin/bash

SRCDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $SRCDIR/..

TOP=$HOME/data/2015-09-FMO_conf/4BCL_pp.top                     \
ATOMS=99548                                                     \
TRJLEN=201                                                      \
./cdctraj.sh $HOME/data/2015-07-FmoEquil/md_100ps_50ns.gro       \
     > $HOME/data/2015-07-FmoEquil/md_100ps_20ns_cdc.txt
