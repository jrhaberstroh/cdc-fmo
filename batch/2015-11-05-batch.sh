#!/bin/bash

cd ..

TOP=$HOME/data/2015-09-FMO_conf/4BCL_pp.top                     \
ATOMS=99548                                                     \
TRJLEN=100                                                      \
./cdctraj.sh $HOME/data/2015-07-FmoEquil/md_100ps_50ns.gro      \
    > $HOME/data/2015-07-FmoEquil/md_100ps_10ns_cdc.txt
