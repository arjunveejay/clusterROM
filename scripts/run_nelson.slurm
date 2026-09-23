#!/bin/bash
#SBATCH -J simple_nelson
# Logs are written under the repository root; submit from there so the relative path resolves.
#SBATCH -o logs/simple_nelson_%j.out
#SBATCH -e logs/simple_nelson_%j.err
#SBATCH -N 1
#SBATCH -n 1                             # one launching task; Python distributes the work over the cores

# Runs the Nelson sweep into a directory of its own, leaving any existing run
# in place so that the two can be compared. The grid:

#   taus  0.01 0.10 0.40
#   etas  1e-4 .. 1e-12
#   ranks 1,3,..,13   (rho = 14)
#   n_eval 2048 testing / 4096 training

# Environment variables, all optional:
#   PKG          the repository root      (default: the parent of this script)
#   DATASETS     where the datasets live  (default: $PKG/../datasets)
#   VENV         virtualenv to activate   (default: none)
#   OUT_DIR      where this run writes    (default: <REF>_check)
#   N_WORKERS    process-pool size        (default: cores on the node)
#   EXTRA_ARGS   passed through verbatim, last, so it can override anything

set -uo pipefail

PKG="${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
# Slurm executes a spooled copy of this script, so the derivation above fails under sbatch unless PKG is exported.
[[ -d "$PKG/src" ]] || { echo "set PKG to the repository root" >&2; exit 2; }
DATASETS="${DATASETS:-$(cd "$PKG/.." && pwd)/datasets}"
REF="$PKG/experiments/nelson/R0.1_M6.0"
OUT_DIR="${OUT_DIR:-${REF}_check}"
DATA="$DATASETS/grav_collapse/nelson/piecewise_constant/R0.1_M6_trace_cells.npy"

mkdir -p "$PKG/logs" "$OUT_DIR"
[[ -n "${VENV:-}" ]] && source "$VENV/bin/activate"

# Keep the math libraries single-threaded per worker, so the process pool rather than BLAS owns the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS="${N_WORKERS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Read-only reuse of REF's feature matrices: open_feature_matrix mmaps them with mode "r", and nothing writes them unless --build-features is passed.
for f in feature_matrix.npy feature_matrix_indices.npy \
         feature_matrix_test.npy feature_matrix_test_indices.npy; do
    [[ -e "$REF/$f" ]] || { echo "missing $REF/$f" >&2; exit 2; }
    ln -sfn "$REF/$f" "$OUT_DIR/$f"
done

TAUS="0.01 0.10 0.40"
ETAS="1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12"
RANKS="1 3 5 7 9 11 13"

cd "$PKG/scripts"
echo "host=$(hostname) cores=$(nproc) n_workers=$N_WORKERS out=$OUT_DIR"

# The testing pass runs first, since it fits every combination; the training pass reloads those fits from disk and only re-evaluates.
for SPLIT in test train; do
    case "$SPLIT" in
        test)  N_EVAL=2048 ;;
        train) N_EVAL=4096 ;;
    esac
    echo "=== nelson ($SPLIT): starting $(date) ==="
    python run_nelson.py \
        --model R0.1_M6.0 --out-dir "$OUT_DIR" --chemical-hydro-file "$DATA" \
        --taus $TAUS --etas $ETAS --ranks $RANKS \
        --n-eval "$N_EVAL" --eval-split "$SPLIT" \
        --n-workers "$N_WORKERS" --no-figures ${EXTRA_ARGS}
    rc=$?
    echo "=== nelson ($SPLIT): exit $rc at $(date) ==="
    (( rc == 0 )) || exit $rc
done
