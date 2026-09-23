#!/bin/bash
#SBATCH -J simple_osu
# Logs are written under the repository root; submit from there so the relative path resolves.
#SBATCH -o logs/simple_osu_%j.out
#SBATCH -e logs/simple_osu_%j.err
#SBATCH -N 1
#SBATCH -n 1                             # one launching task; Python distributes the work over the cores

# Runs the OSU sweep into a directory of its own, leaving any existing run in
# place so that the two can be compared. The grid:
#
#   taus  0.005 0.01 0.10 0.40
#   etas  1e-5 .. 1e-12, then 1e-14 (1e-13 omitted)
#   ranks 1,3,..,29   (rho = 29)
#   n_eval 2048 testing / 4096 training
#
# The feature matrices are symlinked from REF rather than rebuilt: they carry
# the training/testing partition, so reusing them is what makes two runs
# comparable.
#
#     sbatch scripts/run_osu.slurm
#     EXTRA_ARGS="--taus 0.40 --etas 1e-8" sbatch scripts/run_osu.slurm    # short test
#
# Allocation, partition and walltime are site-specific, so they are not in the
# header. Pass them at submit time:
#
#     sbatch -A <account> -p <partition> -t 24:00:00 scripts/run_osu.slurm
#
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
REF="$PKG/experiments/osu/R0.1_M6.0"
OUT_DIR="${OUT_DIR:-${REF}_check}"
DATA="$DATASETS/grav_collapse/osu_09_2008/piecewise_constant/R0.1_M6_trace_cells.npy"

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

TAUS="0.005 0.01 0.10 0.40"
ETAS="1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12 1e-14"
RANKS="1 3 5 7 9 11 13 15 17 19 21 23 25 27 29"

cd "$PKG/scripts"
echo "host=$(hostname) cores=$(nproc) n_workers=$N_WORKERS out=$OUT_DIR"

# The testing pass runs first, since it fits every combination; the training pass reloads those fits from disk and only re-evaluates.
for SPLIT in test train; do
    case "$SPLIT" in
        test)  N_EVAL=2048 ;;
        train) N_EVAL=4096 ;;
    esac
    echo "=== osu ($SPLIT): starting $(date) ==="
    python run_osu.py \
        --model R0.1_M6.0 --out-dir "$OUT_DIR" --chemical-hydro-file "$DATA" \
        --taus $TAUS --etas $ETAS --ranks $RANKS \
        --n-eval "$N_EVAL" --eval-split "$SPLIT" \
        --n-workers "$N_WORKERS" --no-figures ${EXTRA_ARGS}
    rc=$?
    echo "=== osu ($SPLIT): exit $rc at $(date) ==="
    (( rc == 0 )) || exit $rc
done
