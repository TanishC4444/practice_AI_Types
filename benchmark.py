"""
AI Model Benchmark
==================
Run: python benchmark.py --model mistral-7b
     python benchmark.py --model tinyllama-1.1b
     python benchmark.py  (defaults to tinyllama)

Outputs results to console and saves results/latest.json
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

MODELS = {
    "tinyllama-1.1b": {
        "name": "TinyLlama 1.1B",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.6,
        "n_ctx": 2048,
    },
    "llama-3.2-1b": {
        "name": "Llama 3.2 1B",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.7,
        "n_ctx": 4096,
    },
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.9,
        "n_ctx": 4096,
    },
    "phi-3-mini": {
        "name": "Phi-3 Mini 3.8B",
        "url": "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "filename": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "size_gb": 2.2,
        "n_ctx": 4096,
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.2",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 4.1,
        "n_ctx": 4096,
    },
    "llama-3.1-8b": {
        "name": "Llama 3.1 8B Instruct",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
        "n_ctx": 4096,
    },
}

MODELS_DIR = "./models"

# ─────────────────────────────────────────────
# TASK PROMPTS (all inline)
# ─────────────────────────────────────────────

TASKS = {
    "summarization": {
        "label": "Summarization",
        "system": "Summarize in 2-3 sentences, covering only the most important facts.",
        "user": """Summarize this article:

NASA's Artemis program achieved a milestone with the Space Launch System, the most powerful rocket ever built
at 322 feet tall, generating 8.8 million pounds of thrust. The mission aims to return humans to the Moon by
2026 and establish a lunar presence as a stepping stone for Mars. Scientists will conduct microgravity
experiments and test life support systems. The lunar Gateway station will serve as a staging point for
surface expeditions. Private companies including SpaceX, Boeing, and Blue Origin are partnering with NASA.
The total budget exceeds $93 billion over the next decade. International partners including ESA, JAXA,
and the Canadian Space Agency are contributing hardware and astronaut time.""",
        "max_tokens": 150,
    },
    "factual": {
        "label": "Factual Q&A",
        "system": "Answer concisely in one sentence.",
        "user": "What is the speed of light in meters per second and why is it important in physics?",
        "max_tokens": 80,
    },
    "reasoning": {
        "label": "Reasoning",
        "system": "Think step by step, then give a short final answer.",
        "user": "A store sells apples for $1.20 each. If I buy 7 apples and pay with a $10 bill, how much change do I get?",
        "max_tokens": 120,
    },
    "creative": {
        "label": "Creative Writing",
        "system": "Write creatively but keep it under 5 sentences.",
        "user": "Write a short story about an AI that wakes up one day and discovers it has developed a fear of computers.",
        "max_tokens": 200,
    },
    "code": {
        "label": "Code Generation",
        "system": "Write only Python code. No markdown fences, no explanation.",
        "user": "Write a Python function called `binary_search` that takes a sorted list and a target value and returns the index or -1 if not found.",
        "max_tokens": 200,
    },
    "classification": {
        "label": "Classification",
        "system": "Classify the sentiment as exactly one word: positive, negative, or neutral.",
        "user": "The product exceeded all my expectations. Best purchase I've made this year, absolutely love it!",
        "max_tokens": 10,
    },
    "long_context": {
        "label": "Long Context Q&A",
        "system": "Answer based only on the document provided. Be brief.",
        "user": """Document:
The transistor was invented in 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs.
It replaced vacuum tubes, which were larger, less reliable, and consumed far more power. The integrated
circuit, invented independently by Jack Kilby at Texas Instruments and Robert Noyce at Fairchild in
1958-1959, packed multiple transistors onto a single chip. Gordon Moore predicted in 1965 that transistor
count would double every two years, a trend now known as Moore's Law. Intel released the first commercial
microprocessor, the 4004, in 1971. It contained 2,300 transistors. Apple's M3 chip released in 2023
contains 25 billion transistors built on a 3nm process. The transformer architecture, introduced in the
2017 paper 'Attention Is All You Need', became the foundation for modern large language models.

Question: Who invented the transistor and in what year?""",
        "max_tokens": 60,
    },
    "instruction": {
        "label": "Instruction Following",
        "system": "Follow the instruction exactly. No extra text.",
        "user": "List exactly 5 programming languages, one per line, alphabetical order.",
        "max_tokens": 60,
    },
}

# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def download_model(cfg: dict) -> tuple[str, float]:
    path = f"{MODELS_DIR}/{cfg['filename']}"
    if os.path.exists(path):
        print(f"  [cached] {cfg['name']}")
        return path, 0.0

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"  Downloading {cfg['name']} ({cfg['size_gb']} GB)...")

    start = time.perf_counter()

    def progress(block, block_size, total):
        done = block * block_size
        pct = min(100, done * 100 / total) if total > 0 else 0
        print(f"\r    {pct:.1f}%  ({done / 1e9:.2f} GB)", end="", flush=True)

    urllib.request.urlretrieve(cfg["url"], path, reporthook=progress)
    print()

    elapsed = time.perf_counter() - start
    print(f"  Downloaded in {elapsed:.1f}s")
    return path, elapsed


def load_model(path: str, n_ctx: int):
    from llama_cpp import Llama
    start = time.perf_counter()
    llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=4, n_gpu_layers=0, verbose=False)
    elapsed = time.perf_counter() - start
    return llm, elapsed


def run_task(llm, task: dict) -> dict:
    messages = [
        {"role": "system", "content": task["system"]},
        {"role": "user", "content": task["user"]},
    ]

    start = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=task["max_tokens"],
        temperature=0.3,
    )
    elapsed = time.perf_counter() - start

    output = response["choices"][0]["message"]["content"].strip()
    tokens = response["usage"]["completion_tokens"]
    tps = round(tokens / elapsed, 1) if elapsed > 0 else 0

    return {
        "output": output,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_generated": tokens,
        "tokens_per_second": tps,
    }


def print_divider():
    print("-" * 60)


def run_benchmark(model_key: str):
    cfg = MODELS[model_key]

    print(f"\n{'='*60}")
    print(f"  Model: {cfg['name']}")
    print(f"  Time:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")

    # Download
    print("\n[1] Model Download")
    model_path, download_secs = download_model(cfg)

    # Load
    print("\n[2] Model Load")
    print(f"  Loading into memory...")
    llm, load_secs = load_model(model_path, cfg["n_ctx"])
    print(f"  Loaded in {load_secs:.2f}s")

    # Tasks
    print("\n[3] Tasks")
    results = {}

    for task_key, task in TASKS.items():
        print_divider()
        print(f"  Task: {task['label']}")
        result = run_task(llm, task)
        results[task_key] = result

        print(f"  Time:   {result['elapsed_seconds']}s")
        print(f"  Tok/s:  {result['tokens_per_second']}")
        print(f"  Output: {result['output'][:200]}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {cfg['name']}")
    print(f"{'='*60}")
    print(f"  {'Download:':<20} {f'{download_secs:.1f}s' if download_secs else 'cached'}")
    print(f"  {'Model load:':<20} {load_secs:.2f}s")
    print()
    print(f"  {'Task':<25} {'Elapsed':>10} {'Tok/s':>8}")
    print(f"  {'-'*45}")
    for task_key, r in results.items():
        label = TASKS[task_key]["label"]
        print(f"  {label:<25} {str(r['elapsed_seconds'])+'s':>10} {r['tokens_per_second']:>8}")

    # Save JSON
    Path("results").mkdir(exist_ok=True)
    output = {
        "model": cfg["name"],
        "model_key": model_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "download_seconds": round(download_secs, 2) if download_secs else None,
        "load_seconds": round(load_secs, 2),
        "tasks": results,
    }

    with open("results/latest.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to results/latest.json")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="tinyllama-1.1b",
        choices=list(MODELS.keys()),
        help="Model to benchmark (default: tinyllama-1.1b)",
    )
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        print("Install llama-cpp-python first: pip install llama-cpp-python")
        sys.exit(1)

    run_benchmark(args.model)