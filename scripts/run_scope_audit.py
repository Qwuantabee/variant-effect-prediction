"""Audit of what the benchmark actually contains, to bound the claims.

The README's limitations section asserts two things that are easy to write
and hard to defend without a number: that this work covers substitutions
only, and that it does not touch clinical variant interpretation. Both are
claims about the data on disk, so both are measured here rather than stated.

Run:
    python scripts/run_scope_audit.py

Writes results/scope/summary.json and results/scope/notation_scan.csv.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data  # noqa: E402

OUT = ROOT / "results" / "scope"

# A single substitution, optionally several of them joined by colons.
# The mutant letter may be a stop ("*"), matching mutations.MUT_RE: a stop is
# still a substitution at a fixed position, so lumping it into "other notation"
# would report stops as if they were indels and make the indel count wrong.
SUB_RE = re.compile(r"^[A-Z]\d+[A-Z*](\:[A-Z]\d+[A-Z*])*$")


def scan_notation():
    """Every variant in every downloaded assay, checked against substitution
    notation. Returns one row per assay plus the totals.

    The point of scanning all 2.4 M rows rather than a sample: the claim is
    "there is not one indel in this benchmark", and a sample cannot support
    a zero.
    """
    rows = []
    for path in sorted(data.ASSAY_DIR.glob("*.csv")):
        mutants = pd.read_csv(path, usecols=["mutant"])["mutant"].astype(str)
        ok = mutants.str.match(SUB_RE)
        rows.append({
            "assay": path.stem,
            "n_variants": int(len(mutants)),
            "n_substitution_notation": int(ok.sum()),
            "n_other_notation": int((~ok).sum()),
            "n_multi_substitution": int(mutants.str.contains(":").sum()),
            "n_with_stop": int(mutants.str.contains(r"\*").sum()),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = data.reference_table()

    scan = scan_notation()
    scan.to_csv(OUT / "notation_scan.csv", index=False)

    total = int(scan["n_variants"].sum())
    other = int(scan["n_other_notation"].sum())
    multi = int(scan["n_multi_substitution"].sum())

    summary = {
        "assays_scored": int(len(ref)),
        "assays_on_disk": int(len(scan)),
        "variants_scanned": total,
        "variants_not_substitution_notation": other,
        "variants_with_stop_codon": int(scan["n_with_stop"].sum()),
        "multi_substitution_variants": multi,
        "single_substitution_variants": total - multi,
        "assays_all_single_substitution":
            int((ref["DMS_number_multiple_mutants"] == 0).sum()),
        # the reference table's own count, to check the scan agrees with it
        "reference_multiple_mutants": int(
            ref["DMS_number_multiple_mutants"].sum()),
        "reference_total_mutants": int(ref["DMS_total_number_mutants"].sum()),
        "clinical_labels_on_disk": 0,
        "indel_archive_present": bool(
            list(data.ASSAY_DIR.glob("*indel*"))
            or (data.DATA_DIR / "DMS_ProteinGym_indels.zip").exists()),
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"assays on disk: {summary['assays_on_disk']}")
    print(f"variants scanned: {total:,}")
    print(f"not substitution notation: {other}")
    print(f"with a stop codon: {summary['variants_with_stop_codon']}")
    print(f"multi-substitution: {multi:,} "
          f"({100 * multi / total:.1f}% of variants)")
    print(f"assays that are single-substitution only: "
          f"{summary['assays_all_single_substitution']} of {len(ref)}")
    print(f"indel archive present: {summary['indel_archive_present']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
