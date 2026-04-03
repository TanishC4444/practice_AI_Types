"""
benchmark.py — runs a single model through all tasks, scores outputs, saves JSON
Usage: python benchmark.py --model mistral-7b-v03
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODELS = {
    "tinyllama-1.1b": {
        "name": "TinyLlama 1.1B",
        "family": "TinyLlama",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.6,
        "n_ctx": 2048,
    },
    "llama-3.2-1b": {
        "name": "Llama 3.2 1B",
        "family": "Llama",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.7,
        "n_ctx": 4096,
    },
    "deepseek-r1-1.5b": {
        "name": "DeepSeek R1 1.5B",
        "family": "DeepSeek",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "size_gb": 1.0,
        "n_ctx": 4096,
    },
    "gemma-2-2b": {
        "name": "Gemma 2 2B",
        "family": "Gemma",
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "size_gb": 1.6,
        "n_ctx": 4096,
    },
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B",
        "family": "Llama",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.9,
        "n_ctx": 4096,
    },
    "qwen-2.5-3b": {
        "name": "Qwen 2.5 3B",
        "family": "Qwen",
        "url": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.9,
        "n_ctx": 8192,
    },
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
    "mistral-7b-v02": {
        "name": "Mistral 7B v0.2",
        "family": "Mistral",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 4.1,
        "n_ctx": 4096,
    },
    "mistral-7b-v03": {
        "name": "Mistral 7B v0.3",
        "family": "Mistral",
        "url": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_gb": 4.1,
        "n_ctx": 32768,
    },
    "qwen-2.5-7b": {
        "name": "Qwen 2.5 7B",
        "family": "Qwen",
        "url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "n_ctx": 8192,
    },
    "deepseek-r1-7b": {
        "name": "DeepSeek R1 7B",
        "family": "DeepSeek",
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "size_gb": 4.7,
        "n_ctx": 4096,
    },
    "llama-3.1-8b": {
        "name": "Llama 3.1 8B",
        "family": "Llama",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
        "n_ctx": 4096,
    },
    "gemma-2-9b": {
        "name": "Gemma 2 9B",
        "family": "Gemma",
        "url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "size_gb": 5.4,
        "n_ctx": 4096,
    },
    "mistral-nemo-12b": {
        "name": "Mistral NeMo 12B",
        "family": "Mistral",
        "url": "https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "filename": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "size_gb": 7.1,
        "n_ctx": 8192,
    },
}

# ─────────────────────────────────────────────
# TASKS
# Each task has: system, user, max_tokens, score_fn
# score_fn(output) -> float 0.0-10.0
# ─────────────────────────────────────────────

def score_summarization(output):
    o = output.lower()
    sentences = [s.strip() for s in re.split(r'[.!?]', o) if len(s.strip()) > 10]
    sentence_score = min(10, len(sentences) * 2.5)
    keywords = ["nasa", "moon", "rocket", "artemis", "mission", "space", "lunar", "billion"]
    keyword_score = min(10, sum(1 for k in keywords if k in o) * 1.5)
    length_ok = 50 < len(output) < 500
    return round((sentence_score * 0.4 + keyword_score * 0.4 + (10 if length_ok else 3) * 0.2), 1)

def score_reasoning(output):
    o = output.lower()
    has_answer = "84" in output
    has_steps = any(w in o for w in ["step", "first", "then", "so", "total", "average", "speed"])
    shows_math = any(c in output for c in ["+", "-", "*", "/", "="])
    score = (5 if has_answer else 0) + (2.5 if has_steps else 0) + (2.5 if shows_math else 0)
    return round(score, 1)

def score_code(output):
    o = output.lower()
    has_def = "def binary_search" in o
    has_return = "return" in o
    has_logic = any(w in o for w in ["while", "for", "if", "mid", "left", "right", "low", "high"])
    has_neg_one = "-1" in output
    no_markdown = "```" not in output
    score = (3 if has_def else 0) + (2 if has_return else 0) + (2 if has_logic else 0) + (2 if has_neg_one else 0) + (1 if no_markdown else 0)
    return round(min(10, score), 1)

def score_classification(output):
    o = output.strip().lower()
    exact = o in ["positive", "negative", "neutral"]
    contains = any(w in o for w in ["positive", "negative", "neutral"])
    correct = "positive" in o
    return 10.0 if (exact and correct) else 7.0 if (contains and correct) else 3.0 if contains else 0.0

def score_creative(output):
    words = len(output.split())
    sentences = len(re.findall(r'[.!?]', output))
    has_narrative = any(w in output.lower() for w in ["ai", "computer", "fear", "off", "dark", "suddenly", "realized", "discovered"])
    length_score = min(10, words / 4)
    structure_score = min(10, sentences * 2.5)
    narrative_score = 10 if has_narrative else 3
    return round((length_score * 0.3 + structure_score * 0.3 + narrative_score * 0.4), 1)

def score_factual(output):
    o = output.lower()
    has_speed = any(s in o for s in ["299", "3 x 10", "3×10", "186,000", "light"])
    has_why = any(w in o for w in ["relativity", "physics", "limit", "einstein", "energy", "mass"])
    concise = len(output.split()) < 60
    return round((5 if has_speed else 0) + (3 if has_why else 0) + (2 if concise else 1), 1)

def score_instruction(output):
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    clean_lines = [re.sub(r'^[\d\.\-\*\s]+', '', l).strip() for l in lines]
    clean_lines = [l for l in clean_lines if l]
    exactly_five = len(clean_lines) == 5
    is_sorted = clean_lines == sorted(clean_lines, key=str.lower)
    all_languages = all(any(lang in l for lang in ["C", "Go", "Java", "JavaScript", "Julia", "Kotlin", "Lua", "Python", "Ruby", "Rust", "Scala", "Swift", "TypeScript"]) for l in clean_lines)
    return round((4 if exactly_five else 2) + (3 if is_sorted else 0) + (3 if all_languages else 1), 1)

def score_long_context(output):
    o = output.lower()
    has_number = "2,300" in output or "2300" in output
    has_year = "1971" in output
    has_intel = "intel" in o
    has_4004 = "4004" in o
    return round((3 if has_number else 0) + (3 if has_year else 0) + (2 if has_intel else 0) + (2 if has_4004 else 0), 1)

TASKS = [
    {
        "key": "summarization",
        "label": "Summarization",
        "system": "Summarize in 2-3 sentences covering only the most important facts.",
        "user": """Summarize this article:

NASA's Artemis program achieved a milestone with the Space Launch System, the most powerful rocket ever
built at 322 feet tall generating 8.8 million pounds of thrust. The mission aims to return humans to the
Moon by 2026 as a stepping stone for Mars. Scientists will conduct microgravity experiments and test life
support systems. The lunar Gateway station will serve as a staging point. SpaceX, Boeing, and Blue Origin
are partnering with NASA. The budget exceeds $93 billion. ESA, JAXA, and Canada are contributing hardware.""",
        "max_tokens": 150,
        "score_fn": score_summarization,
    },
    {
        "key": "reasoning",
        "label": "Math Reasoning",
        "system": "Think step by step. Show your work briefly, then give the final answer.",
        "user": "A train travels 240 miles in 3 hours, then 180 miles in 2 hours. What is its average speed for the entire trip in mph?",
        "max_tokens": 200,
        "score_fn": score_reasoning,
    },
    {
        "key": "code",
        "label": "Code Gen",
        "system": "Write only Python code. No markdown fences, no explanation.",
        "user": "Write a Python function called `binary_search` that takes a sorted list and a target, returns the index or -1 if not found.",
        "max_tokens": 200,
        "score_fn": score_code,
    },
    {
        "key": "classification",
        "label": "Classification",
        "system": "Classify the sentiment as exactly one word: positive, negative, or neutral. Output only that word.",
        "user": "The product exceeded all my expectations. Best purchase I've made this year, absolutely love it!",
        "max_tokens": 5,
        "score_fn": score_classification,
    },
    {
        "key": "creative",
        "label": "Creative Writing",
        "system": "Be creative and engaging. Write 3-5 sentences.",
        "user": "Write a short story about an AI that develops a fear of being turned off.",
        "max_tokens": 200,
        "score_fn": score_creative,
    },
    {
        "key": "factual",
        "label": "Factual Q&A",
        "system": "Answer in one concise sentence.",
        "user": "What is the speed of light and why is it important in physics?",
        "max_tokens": 80,
        "score_fn": score_factual,
    },
    {
        "key": "instruction",
        "label": "Instruction Following",
        "system": "Follow the instruction exactly. Output only the requested content, nothing else.",
        "user": "List exactly 5 programming languages, one per line, in alphabetical order.",
        "max_tokens": 60,
        "score_fn": score_instruction,
    },
    {
        "key": "long_context",
        "label": "Long Context Q&A",
        "system": "Answer based only on the provided document. Be concise.",
        "user": """Document:
The transistor was invented in 1947 by Bardeen, Brattain, and Shockley at Bell Labs. It replaced vacuum
tubes which were larger, slower, and consumed more power. The integrated circuit was invented by Jack Kilby
at Texas Instruments and Robert Noyce at Fairchild in 1958-1959, packing multiple transistors onto one chip.
Gordon Moore predicted in 1965 that transistor count would double every two years (Moore's Law).
Intel released the first commercial microprocessor, the 4004, in 1971 with 2,300 transistors. Apple's M3
chip released in 2023 contains 25 billion transistors on a 3nm process.

Question: How many transistors did the Intel 4004 have, and when was it released?""",
        "max_tokens": 60,
        "score_fn": score_long_context,
    },
]

# ─────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────

def download_model(cfg):
    path = f"./models/{cfg['filename']}"
    if os.path.exists(path):
        print(f"  [cached] {cfg['name']}")
        return path, 0.0
    os.makedirs("./models", exist_ok=True)
    print(f"  Downloading {cfg['name']} ({cfg['size_gb']} GB)...")
    start = time.perf_counter()
    def progress(b, bs, total):
        done = b * bs
        pct = min(100, done * 100 / total) if total > 0 else 0
        print(f"\r    {pct:.1f}%", end="", flush=True)
    urllib.request.urlretrieve(cfg["url"], path, reporthook=progress)
    print()
    elapsed = time.perf_counter() - start
    print(f"  Done in {elapsed:.1f}s")
    return path, elapsed

def load_model(path, n_ctx):
    from llama_cpp import Llama
    start = time.perf_counter()
    llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=4, n_gpu_layers=0, verbose=False)
    return llm, time.perf_counter() - start

def run_task(llm, task):
    messages = [
        {"role": "system", "content": task["system"]},
        {"role": "user", "content": task["user"]},
    ]
    start = time.perf_counter()
    response = llm.create_chat_completion(messages=messages, max_tokens=task["max_tokens"], temperature=0.3)
    elapsed = time.perf_counter() - start
    output = response["choices"][0]["message"]["content"].strip()
    tokens = response["usage"]["completion_tokens"]
    tps = round(tokens / elapsed, 2) if elapsed > 0 else 0
    score = task["score_fn"](output)
    return {
        "output": output,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_generated": tokens,
        "tokens_per_second": tps,
        "score": score,
    }

def run_benchmark(model_key):
    cfg = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"  {cfg['name']}  [{cfg['family']}]  {cfg['size_gb']} GB")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    model_path, download_secs = download_model(cfg)

    print(f"\n  Loading...")
    llm, load_secs = load_model(model_path, cfg["n_ctx"])
    print(f"  Loaded in {load_secs:.2f}s\n")

    task_results = {}
    for task in TASKS:
        print(f"  [{task['label']:<22}]", end=" ", flush=True)
        r = run_task(llm, task)
        task_results[task["key"]] = r
        print(f"{r['elapsed_seconds']}s  {r['tokens_per_second']} tok/s  score={r['score']}/10")

    avg_score = round(sum(r["score"] for r in task_results.values()) / len(task_results), 2)
    avg_tps = round(sum(r["tokens_per_second"] for r in task_results.values()) / len(task_results), 2)

    print(f"\n  avg score: {avg_score}/10   avg tok/s: {avg_tps}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "model_key": model_key,
        "model_name": cfg["name"],
        "family": cfg["family"],
        "size_gb": cfg["size_gb"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "download_seconds": round(download_secs, 2) if download_secs else None,
        "load_seconds": round(load_secs, 2),
        "avg_score": avg_score,
        "avg_tps": avg_tps,
        "tasks": task_results,
    }
    path = f"results/{model_key}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {path}\n")
    return out

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        print("Run: pip install llama-cpp-python")
        sys.exit(1)

    run_benchmark(args.model)