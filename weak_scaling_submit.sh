#!/bin/bash

# --------------------------------------------------
# LSF CONFIGURATION
# --------------------------------------------------
#BSUB -q hpcintro
#BSUB -J weak_scaling[0,1,2]
#BSUB -W 04:00
#BSUB -R rusage[mem=4GB] 
#BSUB -R span[ptile=24]
#BSUB -n 72
#BSUB -R "select[model == XeonE5_2650v4]"
#BSUB -oo Output_%J_%I.out
#BSUB -eo Output_%J_%I.err

# --------------------------------------------------
# MODULES
# --------------------------------------------------
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

IMPLEMENTATIONS=(
    implementations/decomp_implementation_send_recv.py
    implementations/decomp_implementation.py
    implementations/sendrecv_implementation.py
)


PROCESS_COUNTS=(1 2 3 4 8 12 16 24 32 48 72)
WEAK_N=300
MAX_ITER=400
i=$LSB_JOBINDEX

for p in "${PROCESS_COUNTS[@]}"; do
    N=$(python3 -c "print(round($WEAK_N * $p**(1/3)))")
    echo "Running ${IMPLEMENTATIONS[$i]} with $p processes with N= $N"
    #mpirun -np $p python3 $imp $STRONG_N $MAX_ITER 1e-9 0 false results/strong_scaling.csv 2 1,2
    echo "Done"
done
