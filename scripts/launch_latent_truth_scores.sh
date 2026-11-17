#!/usr/bin/env bash
# Fill the masked-marginal cache for the latent-truth test.
#
# One ESM-2 650M forward pass per mutated position is the whole cost, and each
# protein's table is independent, so the work is split into four shards that
# run as separate user units. Every command here is well under ten minutes;
# the shards are detached because the *total* is about an hour.
#
# CPU only (CUDA_VISIBLE_DEVICES is emptied below): ESMScorer would otherwise
# pick a GPU by default.
#
# Shards are balanced by *measured* cost, positions x sequence length, not by
# protein count: S22A1 needs 547 positions and SPIKE 201 positions of a
# 1,273-residue protein, so counting proteins would leave one shard holding
# most of the work. The four groups below are within ~11% of each other by that
# measure. Threads are 3 per shard because cost is close to linear in threads
# on this model, so four 3-thread processes beat one 12-thread process on
# wall-clock. Each shard resumes from the per-protein cache, so a restart is
# cheap and nothing is scored twice.
set -u
# Repo root is the parent of this script's directory; the interpreter is the
# one on PATH unless PYTHON is set (e.g. PYTHON=.venv/bin/python).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$REPO" || exit 1

mkdir -p results/latent_truth/_cache

SHARDS=(
  "A0A2Z5U3Z0_9INFA,S22A1_HUMAN,KCNE1_HUMAN,RL40A_YEAST,CCDB_ECOLI"
  "CP2C9_HUMAN,HXK4_HUMAN,P53_HUMAN,Q53Z42_HUMAN,BLAT_ECOLX"
  "HSP82_YEAST,SPIKE_SARS2,PTEN_HUMAN"
  "KCNJ2_MOUSE,OXDA_RHOTO,SRC_HUMAN,CAR11_HUMAN,RASK_HUMAN,DYR_ECOLI,SPG1_STRSG"
)

for i in "${!SHARDS[@]}"; do
  unit="latent-truth-shard$i"
  systemctl --user stop "$unit" 2>/dev/null
  systemd-run --user --collect --unit "$unit" \
    --working-directory "$REPO" \
    -p RuntimeMaxSec=7200 \
    bash -c "PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES= '$PY' scripts/run_latent_truth_test.py --scores-only --threads 3 --proteins '${SHARDS[$i]}' > results/latent_truth/${unit}.log 2>&1"
  echo "started $unit: ${SHARDS[$i]}"
done
