#!/bin/bash
set -o nounset
set -o errexit

errecho () { echo "WARNING: $@" 1>&2; }
errcat () { echo "WARNING: " 1>&2; cat $@ 1>&2; }

TOP=$1
QOP=$2
SRCDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
TEMPBASE=$SRCDIR/temp/mol

# Separate each molecule type into a separate text file with
#   comments and section headers removed
errecho "Splitting molecule types from preprocessed topology..."
molecules=$(cat $TOP | grep -n moleculetype | cut -d':' -f 1)
errecho $molecules
molnum=1
for molline in $molecules; do
    temp_file=$TEMPBASE-temp_$molnum.qop
    out_file=$TEMPBASE-$molnum.qop
    tail -n +$molline $TOP > $temp_file
    catch_lines=$(cat $temp_file | grep -n -e '\[' | cut -d':' -f 1)
    catch_num=$(echo $catch_lines | wc -w)
    if [ $catch_num -le 2 ]; then
        errecho "Last molecule is monatomic, update top2qop to allow this..."
        exit 104
    else
        mol_head=`echo $catch_lines | cut -d ' ' -f1`
        mol_atom_beg=`echo $catch_lines | cut -d ' ' -f2`
        mol_atom_end=`echo $catch_lines | cut -d ' ' -f3`
        NAME=$(cat $temp_file | head -n $mol_atom_beg | grep '^[^\[;]' | awk '{print $1}')
        echo $NAME > $out_file
        cat $temp_file | head -n $mol_atom_end | tail -n +$mol_atom_beg \
            | grep '^ *[0-9]' | sed '/^[ \t]*$/d' >> $out_file
            # >> $out_file
    fi
    rm $temp_file
    let molnum++
done

let molnum--
cat $TOP | sed '1,/molecules/d' | grep '^ *[a-zA-Z]' > $TEMPBASE-all.txt
errcat $TEMPBASE-all.txt

errecho NUMBER OF FILES: $molnum
moltypes=''
for i in `seq 1 $molnum`; do
    moltypes="$moltypes $TEMPBASE-$i.qop"
done

python $SRCDIR/build-qop.py -mol $TEMPBASE-all.txt -moltype $moltypes > $QOP
