"""Оценка на GQA-ru (testdev_balanced, 12 216 вопросов) — ExactMatch как в lmms-eval.

Пример:
    python src/eval_gqa.py --tag zeroshot
    python src/eval_gqa.py --tag lora --adapter outputs/lora-adapter
    python src/eval_gqa.py --tag lora --adapter outputs/lora-adapter --split val   # наш hold-out из train
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

from common import (DATA_DIR, GQA_MAX_PIXELS, RESULTS_DIR, generate_batch, gqa_prompt, load_model_and_processor,
                    norm_exact, norm_relaxed, save_json)


def load_testdev():
    from datasets import load_dataset

    qs = load_dataset("deepvk/GQA-ru", "testdev_balanced_instructions", split="testdev")
    imgs = load_dataset("deepvk/GQA-ru", "testdev_balanced_images", split="testdev")
    id2img = {r["id"]: r["image"].convert("RGB") for r in imgs}
    items = [
        dict(id=r["id"], imageId=r["imageId"], question=r["question"], answer=r["answer"],
             structural=r["types"]["structural"], semantic=r["types"]["semantic"])
        for r in qs
    ]
    return items, lambda it: id2img[it["imageId"]]


def load_val():
    path = DATA_DIR / "gqa_val.jsonl"
    items = [json.loads(line) for line in path.open(encoding="utf-8")]
    cache = {}

    def get_img(it):
        if it["image"] not in cache:
            cache[it["image"]] = Image.open(DATA_DIR / it["image"]).convert("RGB")
        return cache[it["image"]]

    return items, get_img


def summarize(rows):
    def acc(key, rs):
        return round(100 * sum(r[key] for r in rs) / max(1, len(rs)), 2)

    by_s, by_sem = defaultdict(list), defaultdict(list)
    for r in rows:
        by_s[r.get("structural")].append(r)
        by_sem[r.get("semantic")].append(r)
    return {
        "n": len(rows),
        "exact_match": acc("em", rows),
        "relaxed_match": acc("em_relaxed", rows),
        "by_structural": {k: {"n": len(v), "em": acc("em", v)} for k, v in sorted(by_s.items(), key=str)},
        "by_semantic": {k: {"n": len(v), "em": acc("em", v)} for k, v in sorted(by_sem.items(), key=str)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="имя прогона, папка results/<tag>/")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="testdev", choices=["testdev", "val"])
    ap.add_argument("--limit", type=int, default=None, help="первые N вопросов (после фикс. перемешивания)")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    items, get_img = load_testdev() if args.split == "testdev" else load_val()
    if args.limit:
        import random

        random.Random(0).shuffle(items)
        items = items[: args.limit]

    out_dir = RESULTS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.split + (f"_{args.limit}" if args.limit else "")
    preds_path = out_dir / f"gqa_{suffix}_preds.jsonl"
    done = {}
    if preds_path.exists():
        for line in preds_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    # сортировка по размеру картинки уменьшает паддинг внутри батча
    todo.sort(key=lambda it: get_img(it).size[0] * get_img(it).size[1])
    print(f"{len(items)} вопросов, уже готово {len(done)}, осталось {len(todo)}")

    model, processor = load_model_and_processor(args.adapter, max_pixels=GQA_MAX_PIXELS)
    t0 = time.time()
    with preds_path.open("a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(todo), args.batch_size), desc="GQA-ru"):
            batch = todo[i: i + args.batch_size]
            preds = generate_batch(model, processor, [get_img(it) for it in batch],
                                   [gqa_prompt(it["question"]) for it in batch], max_new_tokens=16)
            for it, p in zip(batch, preds):
                r = dict(id=it["id"], question=it["question"], answer=it["answer"], pred=p,
                         structural=it.get("structural"), semantic=it.get("semantic"),
                         em=int(norm_exact(p) == norm_exact(it["answer"])),
                         em_relaxed=int(norm_relaxed(p) == norm_relaxed(it["answer"])))
                done[it["id"]] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

    rows = [done[it["id"]] for it in items]
    metrics = summarize(rows)
    metrics.update(split=args.split, adapter=args.adapter, seconds=round(time.time() - t0, 1))
    save_json(metrics, out_dir / f"gqa_{suffix}_metrics.json")
    print(json.dumps({k: metrics[k] for k in ("n", "exact_match", "relaxed_match")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
