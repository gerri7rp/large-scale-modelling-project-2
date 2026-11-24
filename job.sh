
#!/bin/bash

# --------------------------------------------------
# LSF CONFIGURATION
# --------------------------------------------------
#BSUB -q hpc
#BSUB -W 26:00
#BSUB -M 1GB
#BSUB -R "rusage[mem=4GB] select[model == XeonE5_2650v4]"
#BSUB -oo Output_MB.%J.out
#BSUB -eo Output_MB.%J.err

# --------------------------------------------------
# MODULES
# --------------------------------------------------
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------
export PYTHONPATH="$PWD:$PYTHONPATH"

FILE_PATH="implementations/${SCRIPT_NAME}"

if [ ! -f "$FILE_PATH" ]; then
    echo "ERROR: Implementation not found: $FILE_PATH"
    exit 1
fi

# --------------------------------------------------
# OUTPUT CONFIGURATION
# --------------------------------------------------
RUN_TYPE="${SCRIPT_NAME%.py}"
RUN_DIR="results/raw/${RUN_TYPE}/${RUN_ID}"

mkdir -p "$RUN_DIR"

CSV_PATH="${RUN_DIR}/${CSV_NAME}"

# --------------------------------------------------
# LOG CONFIGURATION
# --------------------------------------------------
echo "Implementation: $FILE_PATH"
echo "MPI processes:  $NP"
echo "N:              $N"
echo "Max iterations: $MAX_ITER"
echo "Tolerance:      $TOLERANCE"
echo "Initial value:  $INIT_VALUE"
echo "Plot:           $PLOT"
echo "Batch ID:       $RUN_ID"
echo "CSV output:     $CSV_PATH"
echo "Solution path:  $SOLUTION_PATH"

# --------------------------------------------------
# RUN
# --------------------------------------------------
mpirun -np "$NP" // "$CPU_PER_RANK" python3 "$FILE_PATH" \
    "$N" "$MAX_ITER" "$TOLERANCE" "$INIT_VALUE" "$PLOT" "$CSV_PATH" "$SOLUTION_PATH" "$CPU_PER_RANK"

STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: Experiment failed with exit code $STATUS"
    exit "$STATUS"
fi

echo "Experiment completed successfully."
