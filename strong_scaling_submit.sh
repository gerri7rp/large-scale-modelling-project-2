#!/bin/bash

# --------------------------------------------------
# LSF CONFIGURATION
# --------------------------------------------------
#BSUB -q hpcintro
#BSUB -W 04:00
#BSUB -J strong_scaling
#BSUB -R rusage[mem=4GB]
#BSUB -n 72
#BSUB -R span[ptile=24]
#BSUB -R "select[model == XeonE5_2650v4]"
#BSUB -oo Output_MB.%J.out
#BSUB -eo Output_MB.%J.err

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
STRONG_N=300
MAX_ITER=400
for imp in "${IMPLEMENTATIONS[@]}"; do
    for p in "${PROCESS_COUNTS[@]}"; do
        echo "Running $imp with $p processes"
        mpirun -np $p python3 $imp $STRONG_N $MAX_ITER 1e-9 0 false results/strong_scaling.csv 2 1,2
    done
done