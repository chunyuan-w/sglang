### Bench one batch
1. disable mla
numactl --physcpubind=0-39 --membind=0 python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 8 --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code  --device cpu --attention-backend torch_native --disable-mla

2. Fix mla torch native backend
numactl --physcpubind=0-39 --membind=0 python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 8 --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code  --device cpu --attention-backend torch_native 

vllm:
https://github.com/chunyuan-w/vllm/tree/chunyuan/pr_enable_cpu

flashinfer:
https://github.com/chunyuan-w/flashinfer/tree/chunyuan/act

### Server mode
Cmd for server side:
1. disable mla
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct    --disable-radix --trust-remote-code --device cpu --attention-backend torch_native --disable-mla --log-requests

2. turn on mla (torch native backend)
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct    --disable-radix --trust-remote-code --device cpu --attention-backend torch_native --log-requests

Cmd for client side:
1. run requests
python3 -m sglang.bench_serving --backend sglang --num-prompts 5

2. run accuracy test on mmlu
```sh
# seems we need to be under this repo to run the below accuracy cmd
cd /home/chunyuan/sglang-dev/sglang/python
python3 -m sglang.test.run_eval --eval-name mmlu --num-examples 64 --port 30000
```
