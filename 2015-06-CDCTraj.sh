#!/bin/bash
set -o nounset
set -o errexit

TOP_DEF=~/Jobs/2015-03-4BCL/FMO_conf/4BCL_pp.top
TRJ_DEF=data/FMO-md-short.gro
TOP=${TOP=$TOP_DEF}
TRJ=${TRJ=$TRJ_DEF}

SRCDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
./top2qop.sh $TOP $SRCDIR/temp/4BCL_AMBER94_ppauto.qop
TRJ_TMP=$SRCDIR/temp/`basename $TRJ`
TRJ_TMP_BASE="${TRJ_TMP%.*}"
TRJ_TMP_SUFF="${TRJ_TMP##*.}"
echo BASE: $TRJ_TMP_BASE
echo SUFFIX: $TRJ_TMP_SUFF

cp $TRJ $TRJ_TMP
split -d --additional-suffix .$TRJ_TMP_SUFF -l 99551 $TRJ_TMP $TRJ_TMP_BASE-

for frame in {00..10}; do
    python cdc-fmo.py -groframe $TRJ_TMP_BASE-$frame.$TRJ_TMP_SUFF \
                      -qtop $SRCDIR/temp/4BCL_AMBER94_ppauto.qop \
                      -qcdc $SRCDIR/data/bcl_cdc.txt
done

