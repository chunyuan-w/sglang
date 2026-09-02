"""
SGLang Reward Benchmark Script

Benchmarks reward-model inference through SGLang's /v1/classify endpoint.
Reward models (Skywork-Reward-V2, internlm2-*-reward, etc) attach a
classification head, so serving reuses the classify route with num_classes=1.

Usage:
    python -m sglang.launch_server --model-path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
        --is-embedding --chunked-prefill-size -1 --disable-radix-cache --device cpu

    python bench_reward.py --batch-size 4 --input-tokens 1024
"""

import argparse
import asyncio

from transformers import AutoTokenizer
from util import (
    BenchmarkConfig,
    generate_text_with_token_count,
    run_benchmark_main,
    run_generic_benchmark,
)

###############################################################################
# CONFIG
###############################################################################
config = BenchmarkConfig()
config.rps_values = [1]
config.duration_secs_values = [60]
config.num_unique_requests = 100
config.distribution = "CONSTANT"
config.profile = False
config.freeze_gc = True

HTTP_URL = "http://localhost:30000/v1/classify"

REWARD_INPUT_TOKENS = 1024
REWARD_MODEL_PATH = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
ITEM_COUNT_VALUES = [4]

# " the" is one token on Qwen/LLaMA/BERT alike.
config.special_replicated_token = " the"


def single_sequence_len(tokenizer) -> int:
    overhead = (
        len(tokenizer("a")["input_ids"])
        - len(tokenizer.encode("a", add_special_tokens=False))
    )
    return REWARD_INPUT_TOKENS + overhead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark reward models via SGLang's /v1/classify API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=REWARD_MODEL_PATH)
    parser.add_argument("--url", default=HTTP_URL)
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=ITEM_COUNT_VALUES,
        help="Prompts scored per request; each value is benchmarked in turn.",
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=REWARD_INPUT_TOKENS,
        help="Tokens per prompt (before special tokens).",
    )
    parser.add_argument("--rps", type=int, nargs="+", default=config.rps_values)
    parser.add_argument(
        "--duration", type=int, nargs="+", default=config.duration_secs_values
    )
    parser.add_argument(
        "--distribution",
        choices=["CONSTANT", "POISSON"],
        default=config.distribution,
    )
    parser.add_argument(
        "--num-unique-requests", type=int, default=config.num_unique_requests
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-num-steps", type=int, default=config.profile_num_steps
    )
    parser.add_argument(
        "--profile-start-step", type=int, default=config.profile_start_step
    )
    parser.add_argument("--profiler-dir", default=config.profiler_dir)
    parser.add_argument(
        "--no-freeze-gc", action="store_false", dest="freeze_gc"
    )
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global HTTP_URL, REWARD_MODEL_PATH, ITEM_COUNT_VALUES, REWARD_INPUT_TOKENS
    HTTP_URL = args.url
    REWARD_MODEL_PATH = args.model
    ITEM_COUNT_VALUES = args.batch_size
    REWARD_INPUT_TOKENS = args.input_tokens
    config.rps_values = args.rps
    config.duration_secs_values = args.duration
    config.distribution = args.distribution
    config.num_unique_requests = args.num_unique_requests
    config.profile = args.profile
    config.profile_num_steps = args.profile_num_steps
    config.profile_start_step = args.profile_start_step
    config.profiler_dir = args.profiler_dir
    config.freeze_gc = args.freeze_gc


###############################################################################
# REQUEST GENERATION
###############################################################################
def create_reward_request_builder():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)

    special_token_count = len(
        tokenizer.encode(config.special_replicated_token, add_special_tokens=False)
    )
    print(
        f"Special token '{config.special_replicated_token}' produces "
        f"{special_token_count} token(s)"
    )

    max_len = tokenizer.model_max_length
    seq_len = single_sequence_len(tokenizer)
    # Server rejects seq_len == max_len (strict '>' check), so cap at max_len - 1.
    if seq_len >= max_len:
        overhead = seq_len - REWARD_INPUT_TOKENS
        raise ValueError(
            f"--input-tokens={REWARD_INPUT_TOKENS} encodes to {seq_len} tokens "
            f"per prompt; must be strictly less than {REWARD_MODEL_PATH}'s "
            f"context length {max_len}. Keep --input-tokens at or below "
            f"{max_len - overhead - 1}."
        )
    print(f"Tokens per reward-scored prompt: {seq_len} (max {max_len})")

    def generate_text_with_token_count_local(num_toks):
        return generate_text_with_token_count(
            REWARD_MODEL_PATH,
            num_toks,
            config.special_replicated_token,
            tokenizer=tokenizer,
        )

    def build_reward_request(index: int, item_count: int) -> tuple:
        try:
            texts = [
                generate_text_with_token_count_local(REWARD_INPUT_TOKENS)
                for _ in range(item_count)
            ]
            reward_data = {
                "model": REWARD_MODEL_PATH,
                "input": texts if item_count > 1 else texts[0],
            }
            return (index, reward_data)
        except Exception as e:
            print(f"Error building request {index}: {e}")
            return (index, None)

    return build_reward_request


def validate_reward_response(response_data) -> bool:
    if not isinstance(response_data, dict):
        return False
    data = response_data.get("data")
    if not isinstance(data, list) or not data:
        return False
    return all("label" in item and "probs" in item for item in data)


def build_warmup_reward_request() -> dict:
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    warmup_text = generate_text_with_token_count(
        REWARD_MODEL_PATH,
        REWARD_INPUT_TOKENS,
        config.special_replicated_token,
        tokenizer=tokenizer,
    )
    return {
        "model": REWARD_MODEL_PATH,
        "input": [warmup_text, warmup_text, warmup_text],
    }


###############################################################################
# MAIN
###############################################################################
async def run_benchmark(rps, duration_secs, item_count):
    build_request_func = create_reward_request_builder()
    return await run_generic_benchmark(
        rps=rps,
        duration_secs=duration_secs,
        item_count=item_count,
        config=config,
        http_url=HTTP_URL,
        build_request_func=build_request_func,
        response_validator=validate_reward_response,
        api_name="REWARD",
        request_description="reward requests",
    )


async def main():
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    additional_info = {
        "Input tokens per prompt": REWARD_INPUT_TOKENS,
        "Tokens per scored sequence": single_sequence_len(tokenizer),
    }
    await run_benchmark_main(
        config,
        run_benchmark,
        "REWARD",
        HTTP_URL,
        ITEM_COUNT_VALUES,
        additional_info,
        build_warmup_reward_request,
    )


if __name__ == "__main__":
    apply_args(parse_args())
    asyncio.run(main())
