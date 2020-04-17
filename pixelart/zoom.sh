#!/bin/bash

EXEC="tv-triangulation-color"
ZOOM="$1"
INPUT="input/*.png"
OUTDIR="output-x${ZOOM}"

for i in ${INPUT}; do
    echo "------- $i ---- x ${ZOOM} -------"
    BASE=`echo $i | sed 's/.png//' | sed 's/input\///'`
    IFILE="${BASE}.ppm"
    OFILE="${OUTDIR}/${BASE}"
    convert "${i}" "${IFILE}"
    if test ! -d ${OUTDIR}; then
	mkdir ${OUTDIR}
    fi
    ${EXEC} -b ${ZOOM} -D 16 -i "${IFILE}" -S 1 -o ${OFILE}
    rm ${IFILE}
done
