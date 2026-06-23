# NOTE: This script is the template for the environment variables used in the training script.
#       You may copy this file to `env.sh` and modify the variables as needed.

cd $(dirname "$0")

# Basic path settings
export JSON_PATH=/workspace/inputs
export MODEL_DIR=/data/params/af3.bin
export OUTPUT_DIR=/workspace/outputs
export LOG_DIR=/workspace/log
mkdir -p ${LOG_DIR}
# export DO_PROFILE=1 # Uncomment to enable profiling
export CPU_PER_NUMA=$(lscpu | grep socket | cut -d ' ' -f 22)

# Distributed settings
export MASTER_PORT=1896
export MASTER_ADDR=localhost
export PPN=2
export NNODES=1
export NNODE_MPI=$(echo "${NNODES} * ${PPN}" | bc)
# export FI_PROVIDER=verbs # Uncomment if you have IB cards
export CCL_LOG_LEVEL=info
export CCL_ATL_SHM=1
export CCL_ATL_TRANSPORT=ofi
export CCL_WORKER_COUNT=2
export USE_BACKEND=ccl
export AVAIL_THREADS=$(echo "${CPU_PER_NUMA} - ${CCL_WORKER_COUNT}" | bc)

# Single node settings
export NCORES=${CPU_PER_NUMA}
export RANK=0
