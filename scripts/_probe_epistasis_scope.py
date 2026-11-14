"""Scratch probe: what the multi-mutant assays cost and whether they parse."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data  # noqa: E402
from variant_effect.mutations import parse_variant  # noqa: E402

ASSAYS = [
    "SPG1_STRSG_Olson_2014",
    "HIS7_YEAST_Pokusaeva_2019",
    "PHOT_CHLRE_Chen_2023",
    "GRB2_HUMAN_Faure_2021",
    "GFP_AEQVI_Sarkisyan_2016",
    "CAPSD_AAV2S_Sinai_2021",
]

ref = data.reference_table()
for a in ASSAYS:
    df = pd.read_csv(data.ASSAY_DIR / f"{a}.csv", usecols=["mutant", "DMS_score"])
    nsub = df["mutant"].str.count(":") + 1
    seq = ref.loc[ref["DMS_id"] == a, "target_seq"].iloc[0]
    vc = nsub.value_counts().sort_index()
    bad = 0
    pos = set()
    for m in df["mutant"].head(3000):
        try:
            subs = parse_variant(m)
        except ValueError:
            bad += 1
            continue
        for s in subs:
            if s.idx >= len(seq) or seq[s.idx] != s.wt:
                bad += 1
                break
            pos.add(s.idx)
    print(f"{a:28s} len={len(seq):4d} " +
          " ".join(f"{k}sub={v}" for k, v in vc.items()) +
          f" bad={bad}/3000 pos={len(pos)}", flush=True)
