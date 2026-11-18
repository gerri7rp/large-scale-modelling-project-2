#!/bin/bash
NP=${1:-4}
RUN=results/profile/$(date +%Y%m%d-%H%M%S)_np$NP
mkdir -p $RUN
mpirun -np $NP sh -c "python -m cProfile -o $RUN/rank_\${OMPI_COMM_WORLD_RANK}.prof implementations/sendrecv_implementation.py"
echo "Saved to $RUN"

gprof2dot -f pstats $RUN/rank_0.prof | dot -Tpng -o $RUN/callgraph_rank_0.png   # call graph
python utils/compare_ranks.py $RUN                                                     # cross-rank plot
tuna -o $RUN/tuna_rank_0 $RUN/rank_0.prof && open $RUN/tuna_rank_0/index.html    # tuna
snakeviz $RUN/rank_0.prof                                                        # snakeviz (last: blocks, Ctrl+C to quit)