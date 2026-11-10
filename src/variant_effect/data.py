"""Loading ProteinGym deep mutational scanning assays.

ProteinGym v1.3 collects 217 DMS substitution assays across 186 proteins and,
more usefully, publishes the per-assay scores of ~97 published predictors. That
second part is what makes it a real benchmark rather than just a pile of data:
a new implementation can be checked against the official numbers for the same
model, which catches the off-by-one and tokenisation mistakes that otherwise
look like "my model is a bit worse than the paper".

Everything is downloaded into data/ and never written to by anything else.
The assay archive is third-party data, so it gets its own directory and no
code is ever run from inside it.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"
RAW = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main"

SUBSTITUTIONS_ZIP = f"{BASE}/DMS_ProteinGym_substitutions.zip"
REFERENCE_CSV = f"{RAW}/reference_files/DMS_substitutions.csv"
OFFICIAL_SPEARMAN = (f"{RAW}/benchmarks/DMS_zero_shot/substitutions/Spearman/"
                     "DMS_substitutions_Spearman_DMS_level.csv")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ASSAY_DIR = DATA_DIR / "proteingym"          # extracted third-party data

# A spread across taxa and assay types, all single-substitution only and
# under 300 residues so the whole benchmark runs in minutes rather than hours.
# Official ESM-2 650M Spearman for each is in the benchmark file, which is
# what validate_against_official() checks against.
DEFAULT_ASSAYS = [
    "IF1_ECOLI_Kelsic_2016",        # 72 aa,  prokaryote, organismal fitness
    "CCDB_ECOLI_Tripathi_2016",     # 101 aa, prokaryote, organismal fitness
    "SUMO1_HUMAN_Weile_2017",       # 101 aa, human,      organismal fitness
    "RL40A_YEAST_Roscoe_2013",      # 128 aa, yeast,      organismal fitness
    "DYR_ECOLI_Thompson_2019",      # 159 aa, prokaryote, organismal fitness
    "NUD15_HUMAN_Suiter_2020",      # 164 aa, human,      expression
    "TPMT_HUMAN_Matreyek_2018",     # 245 aa, human,      expression
    "BLAT_ECOLX_Stiffler_2015",     # 286 aa, prokaryote, organismal fitness
]

# Included separately because it is a known failure case rather than a
# representative one -- see the README.
HARD_ASSAYS = ["R1AB_SARS2_Flynn_2022"]      # 306 aa, virus


def _download(url, dest, desc=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"downloading {desc or dest.name} ...", flush=True)
    r = requests.get(url, timeout=600, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    return dest


def reference_table():
    """Metadata for all 217 assays: sequence, length, taxon, assay type."""
    path = _download(REFERENCE_CSV, DATA_DIR / "DMS_substitutions.csv",
                     "ProteinGym reference table")
    return pd.read_csv(path, encoding="utf-8")


def official_scores():
    """Published per-assay Spearman for every baseline in the benchmark.

    Indexed by DMS id, one column per model. 'ESM2 (650M)' is the column
    this project is trying to reproduce.
    """
    path = _download(OFFICIAL_SPEARMAN,
                     DATA_DIR / "official_spearman_dms_level.csv",
                     "official per-assay Spearman")
    df = pd.read_csv(path)
    return df.set_index(df.columns[0])


def ensure_assays(assay_ids=None):
    """Download the archive once and extract only the assays asked for.

    The zip is 41 MB but expands to about 1 GB, so pulling single members out
    of it is worth the small amount of extra code.
    """
    assay_ids = list(assay_ids or DEFAULT_ASSAYS)
    ASSAY_DIR.mkdir(parents=True, exist_ok=True)

    missing = [a for a in assay_ids if not (ASSAY_DIR / f"{a}.csv").exists()]
    if not missing:
        return

    zip_path = _download(SUBSTITUTIONS_ZIP,
                         DATA_DIR / "DMS_ProteinGym_substitutions.zip",
                         "ProteinGym substitution assays (41 MB)")

    with zipfile.ZipFile(zip_path) as zf:
        names = {Path(n).name: n for n in zf.namelist() if n.endswith(".csv")}
        for assay in missing:
            key = f"{assay}.csv"
            if key not in names:
                raise KeyError(f"{assay} is not in the archive. "
                               f"Check the id against reference_table().")
            with zf.open(names[key]) as src:
                (ASSAY_DIR / key).write_bytes(src.read())
            print(f"  extracted {key}", flush=True)


def load_assay(assay_id, reference=None, singles_only=True):
    """One assay as (dataframe, target_sequence).

    The dataframe has columns mutant, mutated_sequence, DMS_score and
    DMS_score_bin. DMS_score is already direction-normalised so that higher
    always means fitter, which is worth knowing because the raw assays point
    in both directions depending on whether the readout was growth or death.
    """
    ensure_assays([assay_id])
    df = pd.read_csv(ASSAY_DIR / f"{assay_id}.csv")

    if singles_only:
        df = df[~df["mutant"].str.contains(":")].reset_index(drop=True)

    ref = reference if reference is not None else reference_table()
    row = ref.loc[ref["DMS_id"] == assay_id]
    if row.empty:
        raise KeyError(f"{assay_id} not found in the reference table")
    sequence = row.iloc[0]["target_seq"]

    return df, sequence


def assay_metadata(assay_id, reference=None):
    ref = reference if reference is not None else reference_table()
    row = ref.loc[ref["DMS_id"] == assay_id]
    if row.empty:
        raise KeyError(f"{assay_id} not found in the reference table")
    r = row.iloc[0]
    return {
        "DMS_id": assay_id,
        "uniprot": r["UniProt_ID"],
        "taxon": r["taxon"],
        "organism": r["source_organism"],
        "seq_len": int(r["seq_len"]),
        "n_singles": int(r["DMS_number_single_mutants"]),
        "selection": r["coarse_selection_type"],
        # deliberately not called "assay": these dicts get merged into result
        # rows that already carry an "assay" column holding the DMS id, and
        # the collision silently replaced the id with this description
        "assay_description": r["selection_assay"],
    }
