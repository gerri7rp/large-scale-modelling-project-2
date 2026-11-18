
#!/bin/bash

# --------------------------------------------------
# LSF CONFIGURATION
# --------------------------------------------------
#BSUB -q hpcintro
#BSUB -W 04:00
#BSUB -M 1500MB
#BSUB -R "select[model == XeonE5_2650v4]"
#BSUB -oo Output_MB.%J.out
#BSUB -eo Output_MB.%J.err

# --------------------------------------------------
# MODULES
# --------------------------------------------------
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

# --------------------------------------------------
# IMPLEMENTATIONS
# --------------------------------------------------
IMPLEMENTATIONS=(
    sendrecv_implementation.py
)

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
PROCESS_COUNTS=(8)
N_VALUES=(10)
MAX_ITER_VALUES=(1000)
TOLERANCE_VALUES=(10e-6)
INIT_VALUE_VALUES=(10)
PLOT_VALUES=(True)

# --------------------------------------------------
# EXPERIMENT IDENTIFIER
# --------------------------------------------------
RUN_ID="$(date +%Y-%m-%d_%H%M)"

BASE_DIR="results/raw"

mkdir -p "$BASE_DIR"

echo "Experiment batch: $RUN_ID"
echo "Submitting jobs..."

# --------------------------------------------------
# SUBMIT ALL EXPERIMENTS
# --------------------------------------------------
TOTAL=0

for script in "${IMPLEMENTATIONS[@]}"; do

    RUN_TYPE="${script%.py}"
    RUN_DIR="${BASE_DIR}/${RUN_TYPE}/${RUN_ID}"
    LOG_DIR="results/logs/${RUN_ID}/${RUN_TYPE}"

    mkdir -p "$RUN_DIR"
    mkdir -p "$LOG_DIR"

    for np in "${PROCESS_COUNTS[@]}"; do
        for n in "${N_VALUES[@]}"; do
            for max_iter in "${MAX_ITER_VALUES[@]}"; do
                for tolerance in "${TOLERANCE_VALUES[@]}"; do
                    for init_value in "${INIT_VALUE_VALUES[@]}"; do
                        for plot in "${PLOT_VALUES[@]}"; do

                            CSV_NAME="np${np}_N${n}_iter${max_iter}_tol${tolerance}_init${init_value}_plot${plot}.csv"
                            LOG_NAME="np${np}_N${n}_iter${max_iter}_tol${tolerance}_init${init_value}_plot${plot}"

                            CSV_PATH="${RUN_DIR}/${CSV_NAME}"

                            echo "Submitting:"
                            echo "  Implementation: $script"
                            echo "  Processes:      $np"
                            echo "  N:              $n"
                            echo "  Max iterations: $max_iter"
                            echo "  Tolerance:      $tolerance"
                            echo "  Initial value:  $init_value"
                            echo "  Plot:           $plot"
                            echo "  Output:         $CSV_PATH"

                            bsub \
                                -J "numerical" \
                                -n "$np" \
                                -R "span[ptile=24]" \
                                -oo "${LOG_DIR}/${LOG_NAME}.out" \
                                -eo "${LOG_DIR}/${LOG_NAME}.err" \
                                -env "SCRIPT_NAME=$script,NP=$np,N=$n,MAX_ITER=$max_iter,TOLERANCE=$tolerance,INIT_VALUE=$init_value,PLOT=$plot,RUN_ID=$RUN_ID,CSV_NAME=$CSV_NAME" \
                                < job.sh

                            if [ $? -eq 0 ]; then
                                TOTAL=$((TOTAL + 1))
                            else
                                echo "ERROR: Failed to submit $script"
                            fi

                        done
                    done
                done
            done
        done
    done
done

echo ""
echo "Submitted $TOTAL jobs."
echo "Results directory: $BASE_DIR/<implementation>/$RUN_ID/"
