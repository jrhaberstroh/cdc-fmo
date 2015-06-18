#!/bin/bash
set -o nounset
set -o errexit

TOP=${TOP=~/Jobs/2015-03-4BCL/FMO_conf_CHARMM27/4BCL_pp.top}
TRJ=${1=UNSET}
TRJLEN=10
TRJLEN=$(echo $TRJLEN - 1 | bc)
ATOMS=99548
ATOMS=$(echo $ATOMS + 3 | bc)

if [ "TRJ" == "UNSET" ]; then
    echo "No gro trajectory passed, use one!"
    exit 1
fi

SRCDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
$SRCDIR/top2qop.sh $TOP $SRCDIR/temp/4BCL_AMBER94_ppauto.qop
TRJ_TMP=$SRCDIR/temp/`basename $TRJ`
TRJ_TMP_BASE="${TRJ_TMP%.*}"
TRJ_TMP_SUFF="${TRJ_TMP##*.}"

cp $TRJ $TRJ_TMP
split -d --additional-suffix .$TRJ_TMP_SUFF -l $ATOMS $TRJ_TMP $TRJ_TMP_BASE-

for frame in $(seq -f %02g $TRJLEN); do
    python $SRCDIR/cdc-fmo.py \
            -groframe $TRJ_TMP_BASE-$frame.$TRJ_TMP_SUFF \
            -qtop $SRCDIR/temp/4BCL_AMBER94_ppauto.qop \
            -qcdc $SRCDIR/data/bcl_cdc.txt
done

