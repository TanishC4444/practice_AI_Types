"""
Multi-Model AI Benchmark
========================
Run locally:   python benchmark.py --model mistral-7b
Run all local: python benchmark.py --all
On Actions:    triggered via workflow_dispatch, one runner per model
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
# MODEL REGISTRY
# All Q4_K_M GGUF — fits in 16GB GitHub runner RAM
# Organized by model family
# ─────────────────────────────────────────────

MODELS = {

    # ── Mistral family ────────────────────────
    "mistral-7b-v02": {
        "name": "Mistral 7B Instruct v0.2",
        "family": "Mistral",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 4.1,
        "n_ctx": 4096,
    },
    "mistral-7b-v03": {
        "name": "Mistral 7B Instruct v0.3",
        "family": "Mistral",
        "url": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_gb": 4.1,
        "n_ctx": 32768,
    },
    "mistral-nemo-12b": {
        "name": "Mistral NeMo 12B Instruct",
        "family": "Mistral",
        "url": "https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "filename": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "size_gb": 7.1,
        "n_ctx": 8192,
    },

    # ── Llama family ──────────────────────────
    "llama-3.2-1b": {
        "name": "Llama 3.2 1B Instruct",
        "family": "Llama",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.7,
        "n_ctx": 4096,
    },
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "family": "Llama",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.9,
        "n_ctx": 4096,
    },
    "llama-3.1-8b": {
        "name": "Llama 3.1 8B Instruct",
        "family": "Llama",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
        "n_ctx": 4096,
    },

    # ── Phi family (Microsoft) ─────────────────
    "phi-3-mini": {
        "name": "Phi-3 Mini 3.8B",
        "family": "Phi",
        "url": "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "filename": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "size_gb": 2.2,
        "n_ctx": 4096,
    },
    "phi-3.5-mini": {
        "name": "Phi-3.5 Mini 3.8B",
        "family": "Phi",
        "url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.2,
        "n_ctx": 4096,
    },
    "phi-4-mini": {
        "name": "Phi-4 Mini 3.8B",
        "family": "Phi",
        "url": "https://huggingface.co/bartowski/phi-4-mini-instruct-GGUF/resolve/main/phi-4-mini-instruct-Q4_K_M.gguf",
        "filename": "phi-4-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.5,
        "n_ctx": 16384,
    },

    # ── Gemma family (Google) ─────────────────
    "gemma-2-2b": {
        "name": "Gemma 2 2B Instruct",
        "family": "Gemma",
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "size_gb": 1.6,
        "n_ctx": 4096,
    },
    "gemma-2-9b": {
        "name": "Gemma 2 9B Instruct",
        "family": "Gemma",
        "url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "size_gb": 5.4,
        "n_ctx": 4096,
    },

    # ── Qwen family (Alibaba) ─────────────────
    "qwen-2.5-3b": {
        "name": "Qwen 2.5 3B Instruct",
        "family": "Qwen",
        "url": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.9,
        "n_ctx": 8192,
    },
    "qwen-2.5-7b": {
        "name": "Qwen 2.5 7B Instruct",
        "family": "Qwen",
        "url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "n_ctx": 8192,
    },

    # ── DeepSeek Reasoning ────────────────────
    "deepseek-r1-1.5b": {
        "name": "DeepSeek R1 Distill 1.5B",
        "family": "DeepSeek",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "size_gb": 1.0,
        "n_ctx": 4096,
    },
    "deepseek-r1-7b": {
        "name": "DeepSeek R1 Distill 7B",
        "family": "DeepSeek",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "size_gb": 4.7,
        "n_ctx": 4096,
    },

    # ── TinyLlama (speed baseline) ────────────
    "tinyllama-1.1b": {
        "name": "TinyLlama 1.1B Chat",
        "family": "TinyLlama",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.6,
        "n_ctx": 2048,
    },
}

# ─────────────────────────────────────────────
# TASKS (all inline)
# ─────────────────────────────────────────────

TASKS = {
    "summarization": {
        "label": "Summarization",
        "system": "Summarize in 2-3 sentences covering only the most important facts.",
        "user": """Summarize this:

NASA's Artemis program achieved a milestone with the Space Launch System, the most powerful rocket ever
built at 322 feet tall generating 8.8 million pounds of thrust. The mission aims to return humans to the
Moon by 2026 as a stepping stone for Mars. Scientists will conduct microgravity experiments and test life
support systems. The lunar Gateway station will serve as a staging point. SpaceX, Boeing, and Blue Origin
are partnering with NASA. The budget exceeds $93 billion. ESA, JAXA, and Canada are contributing hardware.""",
        "max_tokens": 150,
    },
    "reasoning": {
        "label": "Math Reasoning",
        "system": "Think step by step, give a short final answer.",
        "user": "A train travels 240 miles in 3 hours, then 180 miles in 2 hours. What is its average speed for the entire trip?",
        "max_tokens": 150,
    },
    "code": {
        "label": "Code Generation",
        "system": "Write only Python code. No markdown fences, no explanation.",
        "user": "Write a Python function called `binary_search` that searches a sorted list and returns the index or -1.",
        "max_tokens": 200,
    },
    "classification": {
        "label": "Classification",
        "system": "Classify the sentiment as exactly one word: positive, negative, or neutral.",
        "user": "The product exceeded all my expectations. Best purchase I've made this year!",
        "max_tokens": 10,
    },
    "creative": {
        "label": "Creative Writing",
        "system": "Be creative but keep it under 4 sentences.",
        "user": "Write a short story about an AI that develops a fear of being turned off.",
        "max_tokens": 180,
    },
    "factual": {
        "label": "Factual Q&A",
        "system": "Answer in one sentence.",
        "user": "What is the speed of light and why does it matter in physics?",
        "max_tokens": 80,
    },
    "instruction": {
        "label": "Instruction Following",
        "system": "Follow exactly. No extra text.",
        "user": "List exactly 5 programming languages, one per line, in alphabetical order.",
        "max_tokens": 50,
    },
    "long_context": {
        "label": "Long Context Q&A",
        "system": "Answer based only on the document. Be brief.",
        "user": """Document:
The transistor was invented in 1947 by Bardeen, Brattain, and Shockley at Bell Labs. It replaced vacuum
tubes which were larger, slower, and consumed more power. The integrated circuit was invented independently
by Jack Kilby at Texas Instruments and Robert Noyce at Fairchild in 1958-1959, packing multiple transistors
onto one chip. Gordon Moore predicted in 1965 that transistor count would double every two years (Moore's Law).
Intel released the first commercial microprocessor, the 4004, in 1971 with 2,300 transistors. Apple's M3
chip (2023) contains 25 billion transistors on a 3nm process. The transformer architecture from the 2017
paper 'Attention Is All You Need' became the foundation of modern language models.

Question: How many transistors did the Intel 4004 have, and when was it released?""",
        "max_tokens": 60,
    },
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def download_model(cfg: dict) -> tuple[str, float]:
    path = f"./models/{cfg['filename']}"
    if os.path.exists(path):
        print(f"  [cached] {cfg['name']} — skipping download")
        return path, 0.0

    os.makedirs("./models", exist_ok=True)
    print(f"  Downloading {cfg['name']} ({cfg['size_gb']} GB)...")

    start = time.perf_counter()

    def progress(block, block_size, total):
        done = block * block_size
        pct = min(100, done * 100 / total) if total > 0 else 0
        print(f"\r    {pct:.1f}%  ({done / 1e9:.2f} GB)", end="", flush=True)

    urllib.request.urlretrieve(cfg["url"], path, reporthook=progress)
    print()
    elapsed = time.perf_counter() - start
    print(f"  Done in {elapsed:.1f}s")
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
    response = llm.create_chat_completion(messages=messages, max_tokens=task["max_tokens"], temperature=0.3)
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


# ─────────────────────────────────────────────
# MAIN BENCHMARK
# ─────────────────────────────────────────────

def run_benchmark(model_key: str):
    cfg = MODELS[model_key]

    print(f"\n{'='*64}")
    print(f"  {cfg['name']}  [{cfg['family']}]  {cfg['size_gb']} GB")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*64}")

    model_path, download_secs = download_model(cfg)

    print(f"\n  Loading model...")
    llm, load_secs = load_model(model_path, cfg["n_ctx"])
    print(f"  Loaded in {load_secs:.2f}s")

    print(f"\n  Running {len(TASKS)} tasks...\n")

    task_results = {}
    for task_key, task in TASKS.items():
        print(f"  [{task['label']}]", end=" ", flush=True)
        result = run_task(llm, task)
        task_results[task_key] = result
        print(f"{result['elapsed_seconds']}s  |  {result['tokens_per_second']} tok/s")

    # Print summary
    print(f"\n{'─'*64}")
    print(f"  RESULTS: {cfg['name']}")
    print(f"{'─'*64}")
    print(f"  Download : {'cached' if not download_secs else f'{download_secs:.1f}s'}")
    print(f"  Load     : {load_secs:.2f}s")
    print()
    print(f"  {'Task':<25} {'Time':>8}  {'Tok/s':>8}  Output preview")
    print(f"  {'─'*25} {'─'*8}  {'─'*8}  {'─'*20}")
    for task_key, r in task_results.items():
        label = TASKS[task_key]["label"]
        preview = r["output"].replace("\n", " ")[:40]
        print(f"  {label:<25} {str(r['elapsed_seconds'])+'s':>8}  {r['tokens_per_second']:>8}  {preview}")

    # Save
    Path("results").mkdir(exist_ok=True)
    output = {
        "model_key": model_key,
        "model_name": cfg["name"],
        "family": cfg["family"],
        "size_gb": cfg["size_gb"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "download_seconds": round(download_secs, 2) if download_secs else None,
        "load_seconds": round(load_secs, 2),
        "tasks": task_results,
    }

    out_path = f"results/{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Also overwrite latest.json
    with open("results/latest.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to {out_path}")
    print(f"{'='*64}\n")
    return output


# ─────────────────────────────────────────────
# COMPARE: reads all result JSONs and prints table
# ─────────────────────────────────────────────

def compare_all():
    files = sorted(Path("results").glob("*.json"))
    files = [f for f in files if f.name != "latest.json"]

    if not files:
        print("No results found. Run benchmark first.")
        return

    all_results = []
    for f in files:
        with open(f) as fp:
            all_results.append(json.load(fp))

    # Sort by family then size
    all_results.sort(key=lambda x: (x.get("family", ""), x.get("size_gb", 0)))

    task_keys = list(TASKS.keys())

    print(f"\n{'='*100}")
    print(f"  COMPARISON TABLE — {len(all_results)} models")
    print(f"{'='*100}")

    # Header
    header = f"  {'Model':<30} {'Family':<10} {'GB':>4}  {'Load':>6}"
    for t in task_keys:
        header += f"  {TASKS[t]['label'][:10]:>10}"
    print(header)
    print(f"  {'─'*30} {'─'*10} {'─'*4}  {'─'*6}" + "  " + "  ".join(["─"*10]*len(task_keys)))

    for r in all_results:
        row = f"  {r['model_name']:<30} {r['family']:<10} {r['size_gb']:>4}  {str(r['load_seconds'])+'s':>6}"
        for t in task_keys:
            task_r = r["tasks"].get(t, {})
            tps = task_r.get("tokens_per_second", "?")
            row += f"  {str(tps)+' t/s':>10}"
        print(row)

    print(f"{'='*100}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model AI benchmark")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", choices=list(MODELS.keys()), help="Run one model")
    group.add_argument("--all", action="store_true", help="Run all models sequentially (local only)")
    group.add_argument("--compare", action="store_true", help="Print comparison table from saved results")
    group.add_argument("--list", action="store_true", help="List all available models")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Key':<20} {'Name':<35} {'Family':<12} {'Size':>6}")
        print(f"{'─'*20} {'─'*35} {'─'*12} {'─'*6}")
        for k, v in MODELS.items():
            print(f"{k:<20} {v['name']:<35} {v['family']:<12} {v['size_gb']:>5} GB")
        sys.exit(0)

    if args.compare:
        compare_all()
        sys.exit(0)

    try:
        from llama_cpp import Llama
    except ImportError:
        print("Run: pip install llama-cpp-python")
        sys.exit(1)

    if args.all:
        for key in MODELS:
            run_benchmark(key)
        compare_all()
    elif args.model:
        run_benchmark(args.model)
    else:
        parser.print_help()