
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
    sequential.py
)

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
### FOR WEAK
PROCESS_COUNTS=(1) #1 2 4 8 12 16 24 32 48 
N_VALUES=()
for p in "${PROCESS_COUNTS[@]}"; do
    N_VALUES+=("$(python3 -c "print(round(300 * $p ** (1/3)))")")
done

### FOR STRONG
# PROCESS_COUNTS=(1 2 3 4 8 12 16 24 32)
# STRONG_N=96
# N_VALUES=()   # same N for every process count
# for p in "${PROCESS_COUNTS[@]}"; do
#     N_VALUES+=("$STRONG_N")
# done
###


MAX_ITER_VALUES=(50 100 200 300 400)
TOLERANCE_VALUES=(10e-15)
INIT_VALUE_VALUES=(0)
PLOT_VALUES=(False)

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
    SOLUTIONS_DIR="results/solutions/${RUN_TYPE}/${RUN_ID}"

    mkdir -p "$RUN_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$SOLUTIONS_DIR"

    for i in "${!PROCESS_COUNTS[@]}"; do
        np="${PROCESS_COUNTS[$i]}"
        n="${N_VALUES[$i]}"
        for max_iter in "${MAX_ITER_VALUES[@]}"; do
                for tolerance in "${TOLERANCE_VALUES[@]}"; do
                    for init_value in "${INIT_VALUE_VALUES[@]}"; do
                        for plot in "${PLOT_VALUES[@]}"; do

                            CSV_NAME="np${np}_N${n}_iter${max_iter}_tol${tolerance}_init${init_value}_plot${plot}.csv"
                            LOG_NAME="np${np}_N${n}_iter${max_iter}_tol${tolerance}_init${init_value}_plot${plot}"
                            SOLUTION_NAME="np${np}_N${n}_iter${max_iter}_tol${tolerance}_init${init_value}_plot${plot}".bin

                            CSV_PATH="${RUN_DIR}/${CSV_NAME}"
                            SOLUTION_PATH="${SOLUTIONS_DIR}/${SOLUTION_NAME}"


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
                                -env "SCRIPT_NAME=$script,NP=$np,N=$n,MAX_ITER=$max_iter,TOLERANCE=$tolerance,INIT_VALUE=$init_value,PLOT=$plot,RUN_ID=$RUN_ID,CSV_NAME=$CSV_NAME,SOLUTION_PATH=$SOLUTION_PATH" \
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

echo ""
echo "Submitted $TOTAL jobs."
echo "Results directory: $BASE_DIR/<implementation>/$RUN_ID/"
