"""
SGLang Rerank Benchmark Script

This script benchmarks SGLang's rerank API performance using HTTP requests.

Current Features:
- HTTP-only implementation (open source compatible)
- Uses /v1/rerank API endpoint directly
- Cross-encoder scoring of one query against a batch of documents
- Configurable RPS, duration, and batch sizes
- Progress tracking and detailed metrics
- Poisson and constant request distributions

Usage:
- Launch the server with the flags cross-encoders require. Chunked prefill
  splits the token batch but XLMRobertaEmbedding rebuilds position ids from the
  full sequence lengths, so a batch over the chunk size dies with a size
  mismatch in roberta.py; radix cache never hits because every pair is a fresh
  sequence:

    python -m sglang.launch_server --model-path BAAI/bge-reranker-base \
        --is-embedding --chunked-prefill-size -1 --disable-radix-cache

- Run: python bench_rerank.py --batch-size 8 --doc-tokens 224
- Every setting below is also a flag; --help lists them with these defaults.
- Each request ranks --batch-size documents against one query.

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
# Defaults for every flag below; override on the command line rather than here.
config = BenchmarkConfig()
config.rps_values = [1]
config.duration_secs_values = [60]
config.num_unique_requests = 100
# CONSTANT keeps arrivals from bunching up, so the scheduler forms one batch per
# request and the profiled shapes stay fixed. POISSON models realistic traffic.
config.distribution = "CONSTANT"
config.profile = False
config.freeze_gc = True  # Enable GC freeze functionality
# Profiler output directory - by default uses present working directory (pwd)
# Override with --profiler-dir.

# HTTP Configuration
HTTP_URL = "http://localhost:30000/v1/rerank"  # Use rerank API directly

# Rerank API Config
# ITEM_COUNT_VALUES determines number of documents per rerank request
RERANK_QUERY_TOKENS = 32
RERANK_DOC_TOKENS = 224
RERANK_MODEL_PATH = "BAAI/bge-reranker-base"
ITEM_COUNT_VALUES = [8]  # Number of documents per request (batch size)

# The BenchmarkConfig default is Qwen's <|im_start|>, which XLM-RoBERTa
# rerankers split into 7 tokens; " the" is one token there and stays linear
# under repetition, so the token budgets above are exact.
config.special_replicated_token = " the"

def pair_sequence_len(tokenizer) -> int:
    """Tokens the model actually sees for one scored pair.

    A cross-encoder joins query and document into one sequence and the
    tokenizer wraps it in special tokens (<s> q </s></s> d </s> for XLM-R), so
    the sequence is longer than the two budgets by that fixed overhead.
    """
    overhead = (
        len(tokenizer("a", "b")["input_ids"])
        - len(tokenizer.encode("a", add_special_tokens=False))
        - len(tokenizer.encode("b", add_special_tokens=False))
    )
    return RERANK_QUERY_TOKENS + RERANK_DOC_TOKENS + overhead

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang's /v1/rerank API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=RERANK_MODEL_PATH, help="Reranker model.")
    parser.add_argument("--url", default=HTTP_URL, help="Rerank endpoint.")
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=ITEM_COUNT_VALUES,
        help="Documents per request; each value is benchmarked in turn.",
    )
    parser.add_argument(
        "--query-tokens", type=int, default=RERANK_QUERY_TOKENS, help="Query tokens."
    )
    parser.add_argument(
        "--doc-tokens", type=int, default=RERANK_DOC_TOKENS, help="Tokens per document."
    )
    parser.add_argument(
        "--rps",
        type=int,
        nargs="+",
        default=config.rps_values,
        help="Requests per second; each value is benchmarked in turn.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        nargs="+",
        default=config.duration_secs_values,
        help="Seconds per run; each value is benchmarked in turn.",
    )
    parser.add_argument(
        "--distribution",
        choices=["CONSTANT", "POISSON"],
        default=config.distribution,
        help="Request arrival pattern.",
    )
    parser.add_argument(
        "--num-unique-requests",
        type=int,
        default=config.num_unique_requests,
        help="Distinct request bodies to pre-build and cycle through.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap the run in /start_profile and /stop_profile on the server.",
    )
    parser.add_argument(
        "--profile-num-steps",
        type=int,
        default=config.profile_num_steps,
        help="Profile only this many forward passes, then let the server stop "
        "itself. Keeps the trace small enough that the op table is quick to "
        "build; unset profiles the whole run.",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=config.profile_start_step,
        help="Skip this many forward passes before profiling starts.",
    )
    parser.add_argument(
        "--profiler-dir",
        default=config.profiler_dir,
        help="Where the server writes traces.",
    )
    parser.add_argument(
        "--no-freeze-gc",
        action="store_false",
        dest="freeze_gc",
        help="Skip freezing the server's garbage collector before the run.",
    )
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    """Fold the parsed flags back into the module-level knobs the builders read."""
    global HTTP_URL, RERANK_MODEL_PATH, ITEM_COUNT_VALUES
    global RERANK_QUERY_TOKENS, RERANK_DOC_TOKENS

    HTTP_URL = args.url
    RERANK_MODEL_PATH = args.model
    ITEM_COUNT_VALUES = args.batch_size
    RERANK_QUERY_TOKENS = args.query_tokens
    RERANK_DOC_TOKENS = args.doc_tokens

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
# REQUEST GENERATION (in parallel)
###############################################################################
def create_rerank_request_builder():
    """Create a rerank request builder function with shared tokenizer."""
    # Load tokenizer once here to verify special token and get precise counts
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)

    # Verify that our special token produces exactly 1 token
    special_token_count = len(
        tokenizer.encode(config.special_replicated_token, add_special_tokens=False)
    )
    print(
        f"Special token '{config.special_replicated_token}' produces "
        f"{special_token_count} token(s)"
    )

    max_len = tokenizer.model_max_length
    pair_len = pair_sequence_len(tokenizer)
    if pair_len > max_len:
        budget = max_len - (pair_len - RERANK_QUERY_TOKENS - RERANK_DOC_TOKENS)
        raise ValueError(
            f"--query-tokens + --doc-tokens = "
            f"{RERANK_QUERY_TOKENS + RERANK_DOC_TOKENS} encodes to {pair_len} tokens "
            f"per pair, over {RERANK_MODEL_PATH}'s maximum sequence length of "
            f"{max_len}. The server would truncate every pair, so the measured "
            f"length would not be the requested one; keep the two budgets at or "
            f"below {budget}."
        )
    print(f"Tokens per scored pair: {pair_len} (max {max_len})")

    def generate_text_with_token_count_local(num_toks):
        """Generate text with precise token count using replicated token."""
        return generate_text_with_token_count(
            RERANK_MODEL_PATH,
            num_toks,
            config.special_replicated_token,
            tokenizer=tokenizer,
        )

    def build_rerank_request(index: int, item_count: int) -> tuple:
        """Build a single rerank request."""
        try:
            # Generate query and documents for rerank API
            query = generate_text_with_token_count_local(RERANK_QUERY_TOKENS)
            documents = [
                generate_text_with_token_count_local(RERANK_DOC_TOKENS)
                for _ in range(item_count)
            ]

            # Return as dict for rerank API format
            rerank_data = {
                "query": query,
                "documents": documents,
                "model": RERANK_MODEL_PATH,
            }
            return (index, rerank_data)

        except Exception as e:
            print(f"Error building request {index}: {e}")
            return (index, None)

    return build_rerank_request


def validate_rerank_response(response_data) -> bool:
    """Validate rerank API response.

    /v1/rerank returns a bare list of RerankResponse objects, not an object
    with a results key, so this checks the list shape rather than a field.
    """
    if not isinstance(response_data, list) or not response_data:
        return False
    return all("score" in item and "index" in item for item in response_data)


def build_warmup_rerank_request() -> dict:
    """Build a warmup request for the rerank API."""
    # Load tokenizer once for warmup generation
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)

    warmup_query = generate_text_with_token_count(
        RERANK_MODEL_PATH,
        RERANK_QUERY_TOKENS,
        config.special_replicated_token,
        tokenizer=tokenizer,
    )
    warmup_documents = [
        generate_text_with_token_count(
            RERANK_MODEL_PATH,
            RERANK_DOC_TOKENS,
            config.special_replicated_token,
            tokenizer=tokenizer,
        )
        for _ in range(3)
    ]

    return {
        "query": warmup_query,
        "documents": warmup_documents,
        "model": RERANK_MODEL_PATH,
    }


###############################################################################
# MAIN
###############################################################################
async def run_benchmark(rps, duration_secs, item_count):
    """Run a single benchmark with the given RPS value."""
    # Create the request builder function with shared tokenizer
    build_request_func = create_rerank_request_builder()

    return await run_generic_benchmark(
        rps=rps,
        duration_secs=duration_secs,
        item_count=item_count,
        config=config,
        http_url=HTTP_URL,
        build_request_func=build_request_func,
        response_validator=validate_rerank_response,
        api_name="RERANK",
        request_description="rerank requests",
    )


async def main():
    """Main function that runs benchmarks for all RPS values."""
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
    additional_info = {
        "Query tokens per request": RERANK_QUERY_TOKENS,
        "Document tokens per document": RERANK_DOC_TOKENS,
        "Tokens per scored pair": pair_sequence_len(tokenizer),
    }

    await run_benchmark_main(
        config,
        run_benchmark,
        "RERANK",
        HTTP_URL,
        ITEM_COUNT_VALUES,
        additional_info,
        build_warmup_rerank_request,
    )


if __name__ == "__main__":
    apply_args(parse_args())
    asyncio.run(main())
