from argparse import ArgumentParser, Namespace
from typing import List, Union

import numpy as np
import torch
from huggingface_hub import login

from optimum.nvidia.pipelines import pipeline
from transformers import pipeline as raw_pipeline
from tqdm import trange


def get_transformers_pipeline(args: Namespace):
    return raw_pipeline(
        model=args.model,
        torch_dtype=torch.float16,
        model_kwargs={
            "device_map": "balanced",
            "max_memory": {0: "20GiB", "cpu": "64GiB"},
        })


def get_trtllm_pipeline(args: Namespace):
    return pipeline(
        model=args.model,
        use_fp8=args.use_fp8,
        use_cuda_graph=args.use_cuda_graph,
        max_batch_size=args.batch_size,
        max_prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        tp=args.tp,
        pp=args.pp,
        gpus_per_node=args.gpus_per_node,
        world_size=args.world_size
    )

def create_prompt_for_length(batch: int, length: int) -> Union[str, List[str]]:
    tokens = ["I"] * length
    tokens = " ".join(tokens)
    if batch == 1:
        return tokens
    return [tokens] * batch

if __name__ == '__main__':
    parser = ArgumentParser("Hugging Face Optimum-Nvidia Pipelines Benchmarking tool")
    parser.add_argument("--token", type=str, help="Hugging Face Hub token to authenticate the request.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs before collecting metrics.")
    parser.add_argument("--repeat", type=int, default=20, help="Number of runs collecting metrics.")
    parser.add_argument("--batch-size", type=int, required=True, help="Size of the batch.")
    parser.add_argument("--prompt-length", type=int, required=True, help="Size of the prompt to use.")
    parser.add_argument("--output-length", type=int, help="Size of the desired output (prompt included).")

    parser.add_argument("--use-transformers", action="store_true", help="Use transformers pipeline as baseline.")
    parser.add_argument("--use-cuda-graph", action="store_true", help="Turn on CUDA Graph.")
    parser.add_argument("--use-fp8", action="store_true", help="Attempt to benchmark in float8 precision.")
    parser.add_argument("--tp", type=int, default=1, help="Degree of tensor parallelism to apply.")
    parser.add_argument("--pp", type=int, default=1, help="Degree of pipeline parallelism to apply.")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of GPUs per node.")
    parser.add_argument("--world-size", type=int, help="Total number of GPUs over all the node.")
    parser.add_argument("--time-to-first-token", action="store_true",
                        help="Indicate we will only generating a single token.")
    parser.add_argument("model", type=str, help="Model's id to use for the benchmark.")

    args = parser.parse_args()

    if args.token:
        login(args.token)

    # Check use case
    if args.time_to_first_token:
        args.max_new_tokens = 1
        args.min_length = args.prompt_length + 1
        args.output_length = args.prompt_length + 1
    else:
        args.min_length = args.output_length
        args.max_new_tokens = args.output_length - args.prompt_length

    prompt = create_prompt_for_length(args.batch_size, args.prompt_length)
    pipe = get_transformers_pipeline(args) if args.use_transformers else get_trtllm_pipeline(args)

    # Warm up
    for _ in trange(args.warmup, desc="Warming up..."):
        _ = pipe(prompt, max_new_tokens=args.max_new_tokens, min_length=args.min_length, use_cache=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Benchmark
    latencies = []
    for _ in trange(args.repeat, desc="Benchmarking..."):
        start.record()
        _ = pipe(prompt, max_new_tokens=args.max_new_tokens, min_length=args.min_length, use_cache=True)
        end.record()
        torch.cuda.synchronize()

        latencies.append(start.elapsed_time(end))

    latencies = np.array(latencies)

    if args.time_to_first_token:
        print(
            "Time-To-First-Token Latency: "
            f"{latencies.mean().astype(np.uint64)} ms "
            f"(+/- {latencies.std().astype(np.uint64)})"
        )
    else:
        num_tokens = (args.batch_size * args.output_length)
        tokens_per_sec = num_tokens / (latencies / 1e3)
        print(
            "Throughput: "
            f"{tokens_per_sec.mean().astype(np.uint64)} tokens/s "
            f"(+/- {tokens_per_sec.std().astype(np.uint64)})"
        )
