import csv
import json
import math
from pathlib import Path

jsonl_path = Path("results/se_after_noise_complete.jsonl")
csv_path = Path("results/se_after_noise_complete.csv")

# Load JSONL
rows = []
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

print("JSONL rows:", len(rows))
print("JSONL keys sample:", sorted(rows[0].keys()) if rows else [])

# Load CSV
with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)

print("CSV rows:", len(csv_rows))
print("CSV columns:", reader.fieldnames)

label_fields = [c for c in reader.fieldnames if c in ("label", "correctness")]
print("Label fields in CSV:", label_fields)

# Compare a few fields between JSONL and CSV
mismatches = 0
for i in range(min(5, len(rows))):
    j = rows[i]
    c = csv_rows[i]
    for k in ("se_before", "se_mean", "se_std", "delta_se", "correctness"):
        if k in j:
            jv = j[k]
            cv = c.get(k)
            if cv is None:
                mismatches += 1
                print("Missing in CSV:", k)
            else:
                try:
                    cvf = float(cv)
                except Exception:
                    cvf = cv
                if isinstance(jv, (int, float)) and isinstance(cvf, (int, float)):
                    if not math.isclose(float(jv), float(cvf), rel_tol=1e-9, abs_tol=1e-9):
                        mismatches += 1
                        print("Mismatch", k, jv, cvf)

print("Sample mismatches:", mismatches)

# AUROC via rank statistic

def auc_rank(y_true, y_score):
    pairs = list(zip(y_score, y_true))
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    pos = [r for r, (_, y) in zip(ranks, pairs) if y == 1]
    n_pos = len(pos)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = sum(pos)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc

se_before = [float(r["se_before"]) for r in rows]
se_mean = [float(r["se_mean"]) for r in rows]
correctness = [int(float(r["correctness"])) for r in rows]
confab = [1 - c for c in correctness]

print("AUC se_before vs correctness(1=correct):", auc_rank(correctness, se_before))
print("AUC se_before vs confab(1=incorrect):", auc_rank(confab, se_before))
print("AUC se_mean vs correctness(1=correct):", auc_rank(correctness, se_mean))
print("AUC se_mean vs confab(1=incorrect):", auc_rank(confab, se_mean))
