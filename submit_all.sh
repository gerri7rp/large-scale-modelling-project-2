#!/bin/bash


# --------------------------------------------------
# LSF CONFIGURATION
# --------------------------------------------------
#BSUB -q hpcintro
#BSUB -W 04:00
#BSUB -M 1500MB
#BSUB -R "select[model == XeonE5_2650v4]"
#BSUB -oo Output_MB_non_blocking.%J.out
#BSUB -eo Output_MB_non_blocking.%J.err

# --------------------------------------------------
# MODULES
# --------------------------------------------------
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8


# --------------------------------------------------
# IMPLEMENTATIONS
# --------------------------------------------------
IMPLEMENTATIONS=(
    # "static_blocking_send_recv.py"           # Gerard
     "static_nonblocking_send_recv.py"        # Gerard
    # "Static_Gatherv_1Send.py"                # Harris
    # "Static_Gatherv_NSend.py"                # Harris
    "dynamic_send_recv_nonblocking.py"       # Marius
    # "dynamic_send_recv_nonblocking_rank0.py" # Marius
)

# ------------------------------------  --------------
# PARAMETERS
# --------------------------------------------------
PROCESS_COUNTS=(48)
CHUNK_SIZES=(100)
IMAGE_SIZES=("110000x110000") # "70000x70000" "90000x90000"
XLIMS=("-2.2:0.75")
YLIMS=("-1.3:1.3")


# PROCESS_COUNTS=(1 2 4 16 24 48)
# CHUNK_SIZES=(1 10 100 1000 5000)
# IMAGE_SIZES=("10x10" "10000x10000")
# XLIMS=("-2.2:0.75" "-2.2:-1.2" "-0.5:0.5")
# YLIMS=("-1.3:1.3" "-1.3:-0.5" "-0.5:0.5")


# --------------------------------------------------
# EXPERIMENT IDENTIFIER
# Unique for this submission batch
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
        for chunk in "${CHUNK_SIZES[@]}"; do
            for img_size in "${IMAGE_SIZES[@]}"; do
                for i in "${!XLIMS[@]}"; do
                    xlim="${XLIMS[$i]}"
                    ylim="${YLIMS[$i]}"

                    CSV_NAME="np${np}_chunk${chunk}_${img_size}_x${i}_y${i}.csv"
                    LOG_NAME="np${np}_chunk${chunk}_${img_size}_x${i}_y${i}"

                    CSV_PATH="${RUN_DIR}/${CSV_NAME}"

                    echo "Submitting:"
                    echo "  Implementation: $script"
                    echo "  Processes:      $np"
                    echo "  Chunk:          $chunk"
                    echo "  Image:          $img_size"
                    echo "  X limits:       $xlim"
                    echo "  Y limits:       $ylim"
                    echo "  Output:         $CSV_PATH"

                    bsub \
                        -J "mandb" \
                        -n "$np" \
                        -R "span[ptile=24]" \
                        -oo "${LOG_DIR}/${LOG_NAME}.out" \
                        -eo "${LOG_DIR}/${LOG_NAME}.err" \
                        -env "SCRIPT_NAME=$script,NP=$np,CHUNK=$chunk,IMG_SIZE=$img_size,XLIM=$xlim,YLIM=$ylim,RUN_ID=$RUN_ID,CSV_NAME=$CSV_NAME" \
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

echo ""
echo "Submitted $TOTAL jobs."
echo "Results directory: $BASE_DIR/<implementation>/$RUN_ID/"