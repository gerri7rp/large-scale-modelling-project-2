#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    python3 -m venv --system-site-packages "$PROJECT_DIR/.venv"
fi
PYTHON="$PROJECT_DIR/.venv/bin/python"
"$PYTHON" -m pip install numba

mkdir -p "$PROJECT_DIR/results/logs"

for N in {500..1000..100}; do
    for DIM in 1 2 3; do
        case "$DIM" in
            1) AXES_LIST=(0 1 2) ;;
            2) AXES_LIST=(0,1 0,2 1,2) ;;
            3) AXES_LIST=(0,1,2) ;;
        esac

        for AXES in "${AXES_LIST[@]}"; do
            AXES_LABEL="${AXES//,/_}"
            JOB_NAME="decomb_N${N}_D${DIM}_A${AXES_LABEL}"
            RESULT_LOC="results/decomb_N${N}_D${DIM}_A${AXES_LABEL}.bin"
            CSV_DEST="results/decomb_N${N}_D${DIM}_A${AXES_LABEL}.csv"

            bsub \
                -q hpcintro \
                -W 04:00 \
                -n 72 \
                -J "$JOB_NAME" \
                -M 4GB \
                -R "select[model == XeonE5_2650v4] rusage[mem=4GB] span[ptile=24]" \
                -oo "$PROJECT_DIR/results/logs/${JOB_NAME}.%J.out" \
                -eo "$PROJECT_DIR/results/logs/${JOB_NAME}.%J.err" <<EOF
#!/bin/bash
set -euo pipefail

PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
source "\$PROJECT_DIR/.venv/bin/activate"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpirun -np 72 "\$VIRTUAL_ENV/bin/python" -c '
import numba
import runpy
import sys

script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
' implementations/decomb_implementation.py \
    "$N" 10000 1e-4 10 False \
    "$RESULT_LOC" "$CSV_DEST" "$DIM" "$AXES"
EOF
            printf 'Submitted %s\n' "$JOB_NAME"
        done
    done
done
