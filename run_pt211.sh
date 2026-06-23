#!/bin/bash
# This script is used to run AlphaFold on a single node.

set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REQUESTED_OUTPUT_DIR=${OUTPUT_DIR:-}

source "${SCRIPT_DIR}/env.sh" || { \
    echo 'Please place your `env.sh` at the repository root.'; \
    echo 'You can refer to `env.sh.example` for the content of `env.sh`.'; \
    exit 1; \
}
cd /pt211/xfold

export USE_DIST=0
NCORES=72
RANK=0
DB_DIR=/data
MODEL_DIR=/data/params
RAW_INPUT_DIR=/pt211/workspace/inputs
PADDED_OUTPUT_ROOT=/pt211/workspace/outputs
UNPADDED_OUTPUT_ROOT=/pt211/workspace/outputs_unpadded
SGL_GSA_OUTPUT_ROOT=/pt211/workspace/outputs_gsa
SGL_TM_OUTPUT_ROOT=/pt211/workspace/outputs_tm
SGL_GSA_TM_OUTPUT_ROOT=/pt211/workspace/outputs_gsa_tm

INPUT_NAME=${INPUT_NAME:-amp_81.txt}

LOG_DIR=/pt211/workspace/log
HMMER_BIN=/pt211/workspace/hmmer/bin
NUMA_NODE=${NUMA_NODE:-0}
MATCH_OP_BENCH_RUNTIME=${MATCH_OP_BENCH_RUNTIME:-True}
PYTHON_LAUNCHER=(python -m torch.backends.xeon.run_cpu --node-id "${NUMA_NODE}" --ninstances 1 --ncores-per-instance "${NCORES}" --rank "${RANK}")

RUN_DATA_PIPELINE=${RUN_DATA_PIPELINE:-False}
RUN_INFERENCE=${RUN_INFERENCE:-True}
PAD_TO_BUCKETS=${PAD_TO_BUCKETS:-True}
AF3_GRID_SELF_ATTENTION_IMPL=${AF3_GRID_SELF_ATTENTION_IMPL:-cpp}
AF3_SELF_ATTENTION_IMPL=${AF3_SELF_ATTENTION_IMPL:-cpp}
AF3_TRIANGLE_MULTIPLICATION_IMPL=${AF3_TRIANGLE_MULTIPLICATION_IMPL:-cpp}
AF3_GATED_LINEAR_UNIT_IMPL=${AF3_GATED_LINEAR_UNIT_IMPL:-cpp}
export AF3_GRID_SELF_ATTENTION_IMPL
export AF3_SELF_ATTENTION_IMPL
export AF3_TRIANGLE_MULTIPLICATION_IMPL
export AF3_GATED_LINEAR_UNIT_IMPL
DEFAULT_RAW_INPUT_JSON=${RAW_INPUT_DIR}/${INPUT_NAME}

if [[ -n "${REQUESTED_OUTPUT_DIR}" ]]; then
    OUTPUT_DIR=${REQUESTED_OUTPUT_DIR}
elif [[ "${AF3_GRID_SELF_ATTENTION_IMPL}" == "sgl" && "${AF3_TRIANGLE_MULTIPLICATION_IMPL}" == "sgl" ]]; then
    OUTPUT_DIR=${SGL_GSA_TM_OUTPUT_ROOT}
elif [[ "${AF3_GRID_SELF_ATTENTION_IMPL}" == "sgl" ]]; then
    OUTPUT_DIR=${SGL_GSA_OUTPUT_ROOT}
elif [[ "${AF3_TRIANGLE_MULTIPLICATION_IMPL}" == "sgl" ]]; then
    OUTPUT_DIR=${SGL_TM_OUTPUT_ROOT}
elif [[ "${PAD_TO_BUCKETS}" == "True" ]]; then
    OUTPUT_DIR=${PADDED_OUTPUT_ROOT}
else
    OUTPUT_DIR=${UNPADDED_OUTPUT_ROOT}
fi

resolve_processed_input_json() {
    local source_json="$1"
    local output_root="${2:-${OUTPUT_DIR}}"
    python - "$source_json" "$output_root" <<'PY'
import json
import pathlib
import string
import sys

source_json = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])

with source_json.open() as f:
    payload = json.load(f)

name = payload.get("name") or source_json.stem
lower_spaceless_name = name.lower().replace(' ', '_')
allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
sanitised_name = ''.join(ch for ch in lower_spaceless_name if ch in allowed_chars)

print(output_dir / sanitised_name / f'{sanitised_name}_data.json')
PY
}

if [[ "${RUN_DATA_PIPELINE}" == "True" ]]; then
    INPUT_JSON_PATH=${INPUT_JSON_PATH:-${DEFAULT_RAW_INPUT_JSON}}
else
    if [[ -z "${PROCESSED_INPUT_JSON:-}" ]]; then
        PROCESSED_INPUT_JSON=$(resolve_processed_input_json "${DEFAULT_RAW_INPUT_JSON}" "${PADDED_OUTPUT_ROOT}")
    fi
    INPUT_JSON_PATH=${INPUT_JSON_PATH:-${PROCESSED_INPUT_JSON}}
fi

if [[ -d "${HMMER_BIN}" ]]; then
    export PATH="${HMMER_BIN}:${PATH}"
fi

if [[ "${MATCH_OP_BENCH_RUNTIME}" == "True" ]]; then
    for preload_lib in "${CONDA_PREFIX:-}/lib/libiomp5.so" "${CONDA_PREFIX:-}/lib/libtcmalloc.so"; do
        if [[ -f "${preload_lib}" && ":${LD_PRELOAD:-}:" != *":${preload_lib}:"* ]]; then
            export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${preload_lib}"
        fi
    done
    PHYS_CORES=$(lscpu -p=CPU,Core,Node | awk -F, -v n="${NUMA_NODE}" '!/^#/ && $3==n { if (!seen[$2]++) printf("%s%s", sep, $1); sep="," }')
    if [[ -n "${PHYS_CORES}" ]]; then
        NUMACTL_ARGS=(--physcpubind="${PHYS_CORES}" --membind="${NUMA_NODE}")
    else
        NUMACTL_ARGS=(-N "${NUMA_NODE}")
    fi
else
    NUMACTL_ARGS=(-N "${NUMA_NODE}")
fi

mkdir -p "${LOG_DIR}"
DEFAULT_LOG_FILE_NAME=${INPUT_NAME}_gsa_${AF3_GRID_SELF_ATTENTION_IMPL}_tm_${AF3_TRIANGLE_MULTIPLICATION_IMPL}_glu_${AF3_GATED_LINEAR_UNIT_IMPL}_sa_${AF3_SELF_ATTENTION_IMPL}_pad_to_buckets_${PAD_TO_BUCKETS}.log
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${LOG_FILE_NAME:-${DEFAULT_LOG_FILE_NAME}}}
mkdir -p "$(dirname "${LOG_FILE}")"


echo "Running AlphaFold on single node for ${INPUT_JSON_PATH}, NCORES=${NCORES}, RANK=${RANK}, grid_self_attention_impl=${AF3_GRID_SELF_ATTENTION_IMPL}, self_attention_impl=${AF3_SELF_ATTENTION_IMPL}, triangle_multiplication_impl=${AF3_TRIANGLE_MULTIPLICATION_IMPL}, gated_linear_unit_impl=${AF3_GATED_LINEAR_UNIT_IMPL}, run_data_pipeline=${RUN_DATA_PIPELINE}, run_inference=${RUN_INFERENCE}, pad_to_buckets=${PAD_TO_BUCKETS}, output_dir=${OUTPUT_DIR}, log_file=${LOG_FILE}, match_op_bench_runtime=${MATCH_OP_BENCH_RUNTIME}, numactl_args=${NUMACTL_ARGS[*]}, python_launcher=${PYTHON_LAUNCHER[*]}, ld_preload=${LD_PRELOAD:-}"
export JAX_PLATFORMS=cpu
export LD_PRELOAD=/usr/local/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

# ===================== PERF DIAG (remove after debugging) =====================
# Dumps everything the torch xeon launcher banner does NOT show, so the fast
# (original) vs slow (new) runs can be compared. Tee'd to LOG_FILE so it is
# captured in the log file (the script's plain echoes go to the terminal only).
{
  echo "===================== RUN_PT211 PERF DIAG START ====================="
  echo "[diag] timestamp            : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[diag] this script          : ${BASH_SOURCE[0]}"
  echo "[diag] SCRIPT_DIR           : ${SCRIPT_DIR}"
  echo "[diag] sourced env.sh       : ${SCRIPT_DIR}/env.sh"
  echo "[diag] cwd (pwd)            : $(pwd)"
  echo "[diag] whoami               : $(whoami)"
  echo "[diag] python               : $(command -v python 2>/dev/null) ($(python --version 2>&1))"
  echo "[diag] python3              : $(command -v python3 2>/dev/null)"
  echo "[diag] CONDA_PREFIX         : ${CONDA_PREFIX:-<unset>}"
  echo "[diag] CONDA_DEFAULT_ENV    : ${CONDA_DEFAULT_ENV:-<unset>}"
  echo "[diag] NCORES               : ${NCORES}"
  echo "[diag] NUMA_NODE            : ${NUMA_NODE}"
  echo "[diag] MATCH_OP_BENCH_RUNTIME: ${MATCH_OP_BENCH_RUNTIME}"
  echo "[diag] PHYS_CORES           : ${PHYS_CORES:-<unset>}"
  echo "[diag] NUMACTL_ARGS (outer) : ${NUMACTL_ARGS[*]}"
  echo "[diag] PYTHON_LAUNCHER      : ${PYTHON_LAUNCHER[*]}"
  echo "[diag] nproc                : $(nproc)"
  echo "[diag] AF3_GRID_SELF_ATTENTION_IMPL     : ${AF3_GRID_SELF_ATTENTION_IMPL}"
  echo "[diag] AF3_SELF_ATTENTION_IMPL          : ${AF3_SELF_ATTENTION_IMPL}"
  echo "[diag] AF3_TRIANGLE_MULTIPLICATION_IMPL : ${AF3_TRIANGLE_MULTIPLICATION_IMPL}"
  echo "[diag] AF3_GATED_LINEAR_UNIT_IMPL       : ${AF3_GATED_LINEAR_UNIT_IMPL}"
  echo "[diag] --- threading / allocator / lib env (sorted) ---"
  env | grep -E '^(OMP_|KMP_|GOMP_|MKL_|DNNL_|IPEX_|TORCH|ATEN|LD_PRELOAD|LD_LIBRARY_PATH|MALLOC|TCMALLOC|JEMALLOC|NUMA|PYTHONPATH|VECLIB|OPENBLAS|GOTO|JAX_)' | sort
  echo "[diag] --- full content of the sourced env.sh ---"
  cat "${SCRIPT_DIR}/env.sh" 2>/dev/null || echo "[diag] (could not read ${SCRIPT_DIR}/env.sh)"
  echo "===================== RUN_PT211 PERF DIAG END ======================="
} 2>&1 | tee -a "${LOG_FILE}"
# =============================================================================

time numactl "${NUMACTL_ARGS[@]}" "${PYTHON_LAUNCHER[@]}" run_alphafold.py \
    --db_dir=${DB_DIR} \
    --jackhmmer_n_cpu=${NCORES} \
    --nhmmer_n_cpu=${NCORES} \
    --run_data_pipeline=${RUN_DATA_PIPELINE} \
    --run_inference=${RUN_INFERENCE} \
    --pad_to_buckets=${PAD_TO_BUCKETS} \
    --json_path=${INPUT_JSON_PATH} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} 2>&1 \
    | tee -a "${LOG_FILE}"
