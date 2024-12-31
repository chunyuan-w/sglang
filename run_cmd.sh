
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so



# Profiling for one batch
numactl --physcpubind=0-39 --membind=0 python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 2 --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code  --device cpu --attention-backend torch_native --max-total-tokens 2048  --disable-mla