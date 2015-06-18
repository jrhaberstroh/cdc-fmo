#!/bin/bash
set -o nounset
set -o errexit

TOP_DEF=${TOP_DEF=~/Jobs/2015-03-4BCL/FMO_conf_CHARMM27/4BCL_pp.top}
TRJ_DEF=data/FMO-md-short.gro

cd ..
TOP=~/Jobs/2015-03-4BCL/FMO_conf/4BCL_pp.top ./cdc-fmo/cdctraj.sh cdc-fmo/data/FMO-md-short.gro

cd cdc-fmo
TOP=~/Jobs/2015-03-4BCL/FMO_conf/4BCL_pp.top ./cdctraj.sh data/FMO-md-short.gro
