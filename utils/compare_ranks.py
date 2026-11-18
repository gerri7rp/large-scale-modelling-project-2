"""Compare tottime per function across MPI ranks.

Usage: python profiling/compare_ranks.py <run_dir> [top_n]
Writes <run_dir>/compare_ranks.png and <run_dir>/compare_ranks.csv
"""
import glob
import os
import pstats
import sys

import matplotlib
matplotlib.use("Agg")  # save to file, don't block on a window
import matplotlib.pyplot as plt
import pandas as pd

run_dir = sys.argv[1]
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

rows = []
for path in sorted(glob.glob(os.path.join(run_dir, "rank_*.prof"))):
    rank = int(os.path.basename(path)[5:-5])
    stats = pstats.Stats(path).stats
    for (file, line, func), (cc, nc, tt, ct, _) in stats.items():
        if file.startswith("<frozen") or func in ("<built-in method marshal.loads>",
                                                   "<built-in method _imp.create_dynamic>",
                                                   "<built-in method _imp.exec_dynamic>"):
            continue  # import machinery noise
        label = func if file == "~" else f"{func} ({os.path.basename(file)}:{line})"
        rows.append({"rank": rank, "func": label, "tottime": tt, "cumtime": ct, "ncalls": nc})

df = pd.DataFrame(rows)
tot = df.groupby(["func", "rank"])["tottime"].sum().unstack(fill_value=0)
top = tot.loc[tot.max(axis=1).nlargest(top_n).index]

# Imbalance = (max - min) / max across ranks, per function
top_csv = top.copy()
top_csv["imbalance_%"] = 100 * (top.max(axis=1) - top.min(axis=1)) / top.max(axis=1)
top_csv.round(4).to_csv(os.path.join(run_dir, "compare_ranks.csv"))

ax = top.iloc[::-1].plot.barh(figsize=(10, 0.5 * top_n + 2), width=0.8)
ax.set_xlabel("tottime [s]")
ax.set_title(f"Top {top_n} functions by tottime, per rank\n{os.path.basename(run_dir.rstrip('/'))}")
ax.legend(title="rank")
plt.tight_layout()
out = os.path.join(run_dir, "compare_ranks.png")
plt.savefig(out, dpi=150)
print(f"  {out}")
print(top_csv.round(3).to_string())