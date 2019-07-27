#!/bin/bash
#source /opt/intel/bin/compilervars.sh intel64
export OMP_NUM_THREADS=1

NOW=$(date +"%m-%d-%Y")
num=2
N=3     # number of qubit
T=1     # number of time step
EPOCH=2    # number of epoch
BATCH=1000 # batch size
DATA=2000  # total data each epoch
LOAD=0  # default to be 0


name='N'$N/'T'$T+$NOW+$num
rm -r results/$name
mkdir -p results/$name
cp ./*.py results/$name/
cp ./*.sh results/$name/
cd results/$name

python -i training.py $N $T $EPOCH $BATCH $DATA $LOAD |& tee output
