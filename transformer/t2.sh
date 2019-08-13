#!/bin/bash
#source /opt/intel/bin/compilervars.sh intel64
export OMP_NUM_THREADS=1

NOW=$(date +"%m-%d-%Y")
num=1
N=4     # number of qubit
T=10    # number of time step
EPOCH=1    # number of epoch #10
BATCH=10 # batch size #1e2, 1e3
DATA=10  # total data each epoch #1e4 3qubit, 1e6 10 qubit
LOAD=0  # default to be 0


name='N'$N/'T'$T+$NOW+$num
rm -r results/$name
mkdir -p results/$name
cp ./*.py results/$name/
cp ./*.sh results/$name/
cd results/$name

#python -i training.py $N $T $EPOCH $BATCH $DATA $LOAD |& tee output
#python -i training.py $N $T $EPOCH $BATCH $DATA $LOAD
#python -i training3.py $N $T $EPOCH $BATCH $DATA $LOAD
python -i dynamics.py $N $T $EPOCH $BATCH $DATA $LOAD
