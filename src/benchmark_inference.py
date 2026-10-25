"""Замер инференса дообученной модели: память GPU и задержка на один запрос -> results/inference.json.

    python src/benchmark_inference.py
"""
from __future__ import annotations

import json
import statistics
import time

import torch
from datasets import load_dataset

from common import ADAPTER_DIR, GQA_MAX_PIXELS, RESULTS_DIR, generate_batch, gqa_prompt, load_model_and_processor, \
    save_json


def timed(fn, n):
    out, ts = None, []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return out, round(statistics.median(ts), 3)


def main():
    imgs = load_dataset("deepvk/GQA-ru", "testdev_balanced_images", split="testdev")
    images = [imgs[i]["image"].convert("RGB") for i in range(16)]
    model, processor = load_model_and_processor(ADAPTER_DIR, max_pixels=GQA_MAX_PIXELS)
    weights_gb = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()

    generate_batch(model, processor, images[:1], [gqa_prompt("Что на картинке?")])  # прогрев
    short, t_short = timed(lambda: generate_batch(model, processor, images[:1], [gqa_prompt("Что на картинке?")]), 5)
    long_, t_long = timed(lambda: generate_batch(model, processor, images[:1],
                                                 ["Опиши подробно, что изображено на картинке."], max_new_tokens=128),
                          3)
    _, t_batch = timed(lambda: generate_batch(model, processor, images, [gqa_prompt("Что на картинке?")] * 16), 3)
    res = {
        "gpu": torch.cuda.get_device_name(0),
        "weights_gb": round(weights_gb, 2),
        "peak_gb_batch16": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "latency_short_answer_s": t_short,
        "latency_description_128tok_s": t_long,
        "batch16_short_s": t_batch,
        "throughput_short_q_per_s": round(16 / t_batch, 1),
        "example_short": short[0],
        "example_description": long_[0],
    }
    save_json(res, RESULTS_DIR / "inference.json")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
