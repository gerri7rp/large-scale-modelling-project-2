#!/bin/bash
set -euo pipefail

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

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if ! python3 -c 'import numba' >/dev/null 2>&1; then
    echo "Numba is not available to the module Python; install it with python3 -m pip install --user numba." >&2
    exit 1
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

IMPLEMENTATIONS=(
    implementations/decomb_implementation_sendrecv.py
    implementations/decomb_implementation.py
    implementations/sendrecv_implementation.py
)


PROCESS_COUNTS=(1 2 3 4 8 12 16 24 32 48 72)
WEAK_N=300
MAX_ITER=400
i="${LSB_JOBINDEX:?This script must be submitted as an LSF job array}"
if (( i < 0 || i >= ${#IMPLEMENTATIONS[@]} )); then
    echo "Invalid LSB_JOBINDEX: $i" >&2
    exit 1
fi

imp="${IMPLEMENTATIONS[$i]}"
RUN_TYPE="${imp##*/}"
RUN_TYPE="${RUN_TYPE%.py}"
CSV_PATH="$PROJECT_DIR/results/weak_scaling_${RUN_TYPE}.csv"
mkdir -p "$PROJECT_DIR/results"
export RUN_ID="${RUN_ID:-weak-${LSB_JOBID:-manual}-${i}}"

run_mpi_script() {
    local processes="$1"
    local script="$2"
    shift 2

    mpirun -np "$processes" python3 -c '
import numba
import runpy
import sys

script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
' "$script" "$@"
}

for p in "${PROCESS_COUNTS[@]}"; do
    N=$(python3 -c 'import sys; print(round(int(sys.argv[1]) * int(sys.argv[2]) ** (1 / 3)))' "$WEAK_N" "$p")
    echo "Running $imp with $p processes and N=$N"
    run_mpi_script "$p" "$imp" \
        "$N" "$MAX_ITER" 1e-9 0 false "$CSV_PATH" 2 1,2
done
