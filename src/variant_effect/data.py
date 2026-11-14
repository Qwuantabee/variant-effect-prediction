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

import numpy as np
import pandas as pd
import requests

BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"
RAW = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main"

SUBSTITUTIONS_ZIP = f"{BASE}/DMS_ProteinGym_substitutions.zip"
MSA_ZIP = f"{BASE}/DMS_msa_files.zip"
AF2_ZIP = f"{BASE}/ProteinGym_AF2_structures.zip"
REFERENCE_CSV = f"{RAW}/reference_files/DMS_substitutions.csv"
OFFICIAL_SPEARMAN = (f"{RAW}/benchmarks/DMS_zero_shot/substitutions/Spearman/"
                     "DMS_substitutions_Spearman_DMS_level.csv")
OFFICIAL_SUMMARY = (f"{RAW}/benchmarks/DMS_zero_shot/substitutions/Spearman/"
                    "Summary_performance_DMS_substitutions_Spearman.csv")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ASSAY_DIR = DATA_DIR / "proteingym"          # extracted third-party data
MSA_DIR = DATA_DIR / "msa"                   # extracted alignments
STRUCTURE_DIR = DATA_DIR / "structures"      # extracted AlphaFold models

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


def official_summary():
    """Published headline and subgroup averages, one row per predictor.

    This is the table that holds the 0.414 headline for ESM-2 650M, along with
    the per-category, per-taxon and per-MSA-depth splits. Having it locally is
    what lets the aggregation rule in aggregate.py be checked rather than
    asserted.
    """
    path = _download(OFFICIAL_SUMMARY,
                     DATA_DIR / "official_summary_spearman.csv",
                     "official summary Spearman")
    return pd.read_csv(path).set_index("Model_name")


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


def canonical_scores(assay_id):
    """Finite DMS_score per variant, keyed by the CANONICAL mutant string.

    Joining two assays on the raw mutant column silently drops every
    multi-substitution variant that the two labs wrote in different orders (a
    709-double loss on the SPG1 Olson/Wu pair; no other of the 37 repo pairs is
    affected). The key here is resampling.canonical_mutant: substitutions sorted
    by position. A variant listed twice after canonicalisation keeps its FIRST
    row; the number dropped is in the returned Series' attrs['duplicates_dropped'].
    An unparseable notation keeps its raw string (counted in
    attrs['unparseable']).
    """
    from .resampling import canonical_mutant
    d = pd.read_csv(ASSAY_DIR / f"{assay_id}.csv",
                    usecols=["mutant", "DMS_score"])
    d = d[np.isfinite(d["DMS_score"])]
    keys, bad = [], 0
    for m in d["mutant"]:
        if ":" not in m:
            keys.append(m)
            continue
        try:
            keys.append(canonical_mutant(m))
        except ValueError:
            keys.append(m)
            bad += 1
    s = pd.Series(d["DMS_score"].to_numpy(), index=pd.Index(keys, name="mutant"))
    dup = int(s.index.duplicated().sum())
    s = s[~s.index.duplicated(keep="first")]
    s.attrs["duplicates_dropped"] = dup
    s.attrs["unparseable"] = bad
    return s


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


def structure_path(assay_id, reference=None):
    """Path to the assay's predicted structure, extracting it on first use.

    ProteinGym distributes AlphaFold2 models for all 217 assays (19 MB packed).
    They are keyed by UniProt id rather than DMS id, and several assays share
    one structure -- the two SPIKE_SARS2 assays are the same protein under two
    readouts, and the three BLAT_ECOLX assays likewise -- so the file is looked
    up by the reference table's ``pdb_file`` column, not by assay.
    """
    ref = reference if reference is not None else reference_table()
    row = ref.loc[ref["DMS_id"] == assay_id]
    if row.empty:
        raise KeyError(f"{assay_id} not found in the reference table")
    name = row.iloc[0]["pdb_file"]
    if not isinstance(name, str) or not name:
        raise KeyError(f"{assay_id} has no structure listed")

    out = STRUCTURE_DIR / name
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out

    archive = _download(AF2_ZIP, DATA_DIR / "ProteinGym_AF2_structures.zip",
                        "AlphaFold2 structures (19 MB)")
    with zipfile.ZipFile(archive) as zf:
        member = next((n for n in zf.namelist() if Path(n).name == name), None)
        if member is None:
            raise KeyError(f"{name} is not in the structure archive")
        out.write_bytes(zf.read(member))
    return out


def msa_path(assay_id, reference=None):
    """Path to the assay's multiple sequence alignment, extracted on first use.

    The archive is 1.5 GB packed and a single member is read by decompressing
    from the start of the file, so extraction is cached on disk. Members are
    keyed by the reference table's ``MSA_filename``, which is not the DMS id --
    the filename carries the alignment's theta and build date.
    """
    ref = reference if reference is not None else reference_table()
    row = ref.loc[ref["DMS_id"] == assay_id]
    if row.empty:
        raise KeyError(f"{assay_id} not found in the reference table")
    name = row.iloc[0]["MSA_filename"]
    if not isinstance(name, str) or not name:
        raise KeyError(f"{assay_id} has no alignment listed")

    out = MSA_DIR / name
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out

    archive = _download(MSA_ZIP, DATA_DIR / "DMS_msa_files.zip",
                        "ProteinGym alignments (1.5 GB)")
    with zipfile.ZipFile(archive) as zf:
        member = next((n for n in zf.namelist() if n.endswith(name)), None)
        if member is None:
            raise KeyError(f"{name} is not in the alignment archive")
        out.write_bytes(zf.read(member))
    return out


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
