#!/bin/bash
# Use the uv venv (Python 3.12); run `myload` first for MPI/numpy modules.
source "$(dirname "$0")/.venv/bin/activate"
NP=${1:-4}
RUN=results/profile/$(date +%Y%m%d-%H%M%S)_np$NP
export MPIP="-d -f $RUN"
# myload is a .bashrc function and is not visible inside scripts, so load the modules here
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
module load mpiP
mkdir -p $RUN

mpirun -np $NP sh -c "python -m cProfile -o $RUN/rank_\${OMPI_COMM_WORLD_RANK}.prof implementations/sendrecv_implementation.py"
echo "Saved to $RUN"
mpirun -np $NP -x LD_PRELOAD=$MPIP_DIR/lib/libmpiP.so -x MPIP python implementations/sendrecv_implementation.py


gprof2dot -f pstats $RUN/rank_0.prof | dot -Tpng -o $RUN/callgraph_rank_0.png   # call graph
python utils/compare_ranks.py $RUN                                                     # cross-rank plot

# Servers (no browser on the cluster). Ports are picked free because the login node is shared.
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
TUNA_PORT=$(free_port)
SNAKE_PORT=$(free_port)
echo "From your laptop: ssh -L $TUNA_PORT:localhost:$TUNA_PORT -L $SNAKE_PORT:localhost:$SNAKE_PORT $(hostname)"
echo "  tuna     http://localhost:$TUNA_PORT"
echo "  snakeviz http://localhost:$SNAKE_PORT  (Ctrl+C to quit)"
tuna --no-browser --port $TUNA_PORT $RUN/rank_0.prof > /dev/null &
trap "kill $!" EXIT
snakeviz -s -H 127.0.0.1 -p $SNAKE_PORT $RUN/rank_0.prof
