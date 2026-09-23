#!/bin/bash
#SBATCH -J simple_lorenz96
# Logs are written under the repository root; submit from there so the relative path resolves.
#SBATCH -o logs/simple_lorenz96_%j.out
#SBATCH -e logs/simple_lorenz96_%j.err
#SBATCH -N 1
#SBATCH -n 1                             # one launching task; Python distributes the work over the cores

# Runs the Lorenz-96 sweep into a directory of its own, leaving any existing
# run in place so that the two can be compared. The grid:

#   taus  0.005 0.01 0.02 0.05 0.25   
#   etas  1e-2 .. 1e-8, then 1e-10    (1e-9 omitted)
#   ranks 2,6,..,98,100               (rho = 100)
#   n_eval 50 testing / 150 training
#

# Environment variables, all optional:
#   PKG          the repository root      (default: the parent of this script)
#   DATASETS     where the datasets live  (default: $PKG/../datasets)
#   VENV         virtualenv to activate   (default: none)
#   OUT_DIR      where this run writes    (default: <REF>_check)
#   GEN_DIR      generator output to read
#   N_WORKERS    process-pool size        (default: cores on the node)
#   EXTRA_ARGS   passed through verbatim, last, so it can override anything

set -uo pipefail

PKG="${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
# Slurm executes a spooled copy of this script, so the derivation above fails under sbatch unless PKG is exported.
[[ -d "$PKG/src" ]] || { echo "set PKG to the repository root" >&2; exit 2; }
DATASETS="${DATASETS:-$(cd "$PKG/.." && pwd)/datasets}"
REF="$PKG/experiments/lorenz96/N100_F0-2.5"
OUT_DIR="${OUT_DIR:-${REF}_check}"
GEN_DIR="${GEN_DIR:-$DATASETS/lorenz96/N100_F0-2.5}"

mkdir -p "$PKG/logs" "$OUT_DIR"
[[ -n "${VENV:-}" ]] && source "$VENV/bin/activate"

# Keep the math libraries single-threaded per worker, so the process pool rather than BLAS owns the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS="${N_WORKERS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Read-only reuse of REF's feature matrices and its shifted source.
for f in feature_matrix.npy feature_matrix_indices.npy \
         feature_matrix_test.npy feature_matrix_test_indices.npy \
         lorenz96_source.npy; do
    [[ -e "$REF/$f" ]] || { echo "missing $REF/$f" >&2; exit 2; }
    ln -sfn "$REF/$f" "$OUT_DIR/$f"
done

TAUS="0.005 0.01 0.02 0.05 0.25"
ETAS="1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-10"
RANKS="2 6 10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98 100"

cd "$PKG/scripts"
echo "host=$(hostname) cores=$(nproc) n_workers=$N_WORKERS out=$OUT_DIR"

# The testing pass runs first, since it fits every combination; the training pass reloads those fits from disk and only re-evaluates.
for SPLIT in test train; do
    case "$SPLIT" in
        test)  N_EVAL=50 ;;
        train) N_EVAL=150 ;;
    esac
    echo "=== lorenz96 ($SPLIT): starting $(date) ==="
    python run_lorenz96.py \
        --out-dir "$OUT_DIR" --gen-dir "$GEN_DIR" \
        --taus $TAUS --etas $ETAS --ranks $RANKS \
        --n-eval "$N_EVAL" --eval-split "$SPLIT" \
        --n-workers "$N_WORKERS" --no-figures ${EXTRA_ARGS}
    rc=$?
    echo "=== lorenz96 ($SPLIT): exit $rc at $(date) ==="
    (( rc == 0 )) || exit $rc
done
