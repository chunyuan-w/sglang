
numactl --physcpubind=0-39 --membind=0 python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 8 --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code  --device cpu --attention-backend torch_native 
--disable-mla

vllm:
https://github.com/chunyuan-w/vllm/tree/chunyuan/pr_enable_cpu