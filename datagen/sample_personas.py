"""
Reservoir sampling from a large JSONL persona file.
Produces a uniformly random subset without loading the full file into memory.
"""
import json
import random

SRC  = "/Users/allenouyang/Project/WORKSPACE/mc_analysis/Elite Personas Part 1.jsonl"
DST  = "/Users/allenouyang/Project/WORKSPACE/mc_analysis/datagen/personas_10k.jsonl"
K    = 10_000
SEED = 42

random.seed(SEED)
reservoir: list[str] = []
i = 0

print(f"Sampling {K:,} personas from {SRC} ...")
with open(SRC, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        if len(reservoir) < K:
            reservoir.append(line)
        else:
            j = random.randint(0, i)
            if j < K:
                reservoir[j] = line
        if (i + 1) % 500_000 == 0:
            print(f"  scanned {i+1:,} lines ...", flush=True)

random.shuffle(reservoir)  # 打乱，避免原文件顺序偏移

with open(DST, "w", encoding="utf-8") as f:
    for line in reservoir:
        f.write(line + "\n")

print(f"\nDone. Scanned {i+1:,} lines → sampled {len(reservoir):,} → {DST}")
