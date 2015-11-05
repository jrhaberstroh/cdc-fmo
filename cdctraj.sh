# cdctraj.sh
#==============================================================================
# Converts a .gro trajectory of length (frames >= TRJLEN) into a series of cdc
# computations. 
#   Usage: 
#     $1       - .gro format trajectory to pass in
#     TOP      - Preprocessed GROMACS topology file
#     ATOMS    - Number of atoms in each frame of $1
#     TRJLEN   - Number of frames to process within gro file ($1). If 
#                successive files repeat the final frame, take care to
#                set TRJLEN to a small number.
#                NOTE: Not error checked to assert that TRJLEN <= numframes

#!/bin/bash
set -o nounset
set -o errexit

SRCDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
CDCFMO_ARGS=${CDCFMO_ARGS= }
CDCFMO_PYARGS=${CDCFMO_PYARGS= }

TOP=${TOP?Input Error: cdctraj.sh requires topology input via TOP envt var}
>&2 echo TRJ used: ${1?ERROR: usage: Pass .gro trajectory as \$1}
TRJ=$1
TRJLEN=${TRJLEN=10}
TRJLEN=$(echo $TRJLEN - 1 | bc)
ATOMS=${ATOMS=99548}
ATOMS=$(echo $ATOMS + 3 | bc)

>&2 echo "WARNING: This program copies $(du -h $TRJ | cut -f1) x 2 data to" \
        "$SRCDIR/temp, make sure this is OK."

$SRCDIR/top2qop.sh $TOP $SRCDIR/temp/4BCL_AMBER94_ppauto.qop
TRJ_TMP=$SRCDIR/temp/`basename $TRJ`
TRJ_TMP_BASE="${TRJ_TMP%.*}"
TRJ_TMP_SUFF="${TRJ_TMP##*.}"

cp $TRJ $TRJ_TMP
# TODO: Split this using trjconv instead of with this awful method
split -d --additional-suffix .$TRJ_TMP_SUFF -l $ATOMS $TRJ_TMP $TRJ_TMP_BASE-


for frame in $(seq -f %02g 00 $TRJLEN); do
    # TODO: Implement measurement of rate-limiting step in script output,
    #       for use on NERSC and here
    if [ ! -e $TRJ_TMP_BASE-$frame.$TRJ_TMP_SUFF ]; then
        echo "ERROR: File \"$TRJ_TMP_BASE-$frame.$TRJ_TMP_SUFF\" does not exist, reduce TRJLEN"
        exit 1
    fi
    python $CDCFMO_PYARGS $SRCDIR/cdc-fmo.py $CDCFMO_ARGS \
            -groframe $TRJ_TMP_BASE-$frame.$TRJ_TMP_SUFF \
            -qtop $SRCDIR/temp/4BCL_AMBER94_ppauto.qop \
            -qcdc $SRCDIR/data/bcl_cdc.txt
done

