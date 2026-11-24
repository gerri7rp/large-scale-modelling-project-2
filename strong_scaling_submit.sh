#!/bin/bash

# --------------------------------------------------
# LSF CONFIGURATION
# --------------------------------------------------
#BSUB -q hpcintro
#BSUB -W 04:00
#BSUB -R rusage[mem=4GB] 
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
    decomp_implementation_send_recv.py
    decomp_implementation.py
    sendrecv_implementation.py
)


PROCESS_COUNTS=(1 2 3 4 8 12 16 24 32 48 72)
STRONG_N=300
MAX_ITER=400
for p in "${PROCESS_COUNTS[@]}"; do
    N_VALUES+=("$STRONG_N")
done