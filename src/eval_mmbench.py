"""Оценка на MMBench-ru (dev, 3 910 записей = 1 109 вопросов с циклическими перестановками вариантов).

Основная метрика — CircularEval, как в lmms-eval (MMBench_Evaluator.eval_result): вопрос засчитывается,
только если модель верно ответила на ВСЕ перестановки его вариантов (index % 1e6 одинаковый),
учитываются группы с исходной записью (index < 1e6). Дополнительно — обычная точность по записям.

Пример:
    python src/eval_mmbench.py --tag zeroshot
    python src/eval_mmbench.py --tag lora --adapter outputs/lora-adapter
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from tqdm import tqdm

from common import (MMBENCH_MAX_PIXELS, OPTION_LETTERS, RESULTS_DIR, extract_letter, generate_batch, is_missing,
                    load_model_and_processor, mc_prompt, save_json)


def load_mmbench():
    from datasets import load_dataset

    ds = load_dataset("deepvk/MMBench-ru", split="dev")
    items = []
    for i, r in enumerate(ds.remove_columns("image")):
        options = {k: r[k] for k in OPTION_LETTERS if k in r and not is_missing(r[k])}
        items.append(dict(row=i, index=int(r["index"]), question=r["question"],
                          hint=None if is_missing(r["hint"]) else r["hint"], options=options,
                          answer=r["answer"], category=r["category"], l2=r["l2-category"]))
    return items, ds


def summarize(rows):
    by_idx = {r["index"]: r for r in rows}
    groups = defaultdict(list)
    for r in rows:
        groups[r["index"] % 1_000_000].append(r)
    circ, circ_cat, circ_l2 = [], defaultdict(list), defaultdict(list)
    for base, grp in groups.items():
        if base not in by_idx:  # как в lmms-eval: считаем только группы с исходной записью
            continue
        hit = int(all(r["correct"] for r in grp))
        circ.append(hit)
        circ_cat[by_idx[base]["category"]].append(hit)
        circ_l2[by_idx[base]["l2"]].append(hit)

    def pct(xs):
        return round(100 * sum(xs) / max(1, len(xs)), 2)

    return {
        "n_rows": len(rows),
        "n_questions": len(circ),
        "circular_acc": pct(circ),
        "vanilla_acc": pct([r["correct"] for r in rows]),
        "unparsed_rate": pct([r["letter"] is None for r in rows]),
        "circular_by_l2": {k: {"n": len(v), "acc": pct(v)} for k, v in sorted(circ_l2.items())},
        "circular_by_category": {k: {"n": len(v), "acc": pct(v)} for k, v in sorted(circ_cat.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="для отладки: первые N записей")
    args = ap.parse_args()

    items, ds = load_mmbench()
    if args.limit:
        items = items[: args.limit]
    out_dir = RESULTS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "dev" + (f"_{args.limit}" if args.limit else "")
    preds_path = out_dir / f"mmbench_{suffix}_preds.jsonl"
    done = {}
    if preds_path.exists():
        for line in preds_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["index"]] = r
    todo = [it for it in items if it["index"] not in done]
    print(f"{len(items)} записей, уже готово {len(done)}, осталось {len(todo)}")

    model, processor = load_model_and_processor(args.adapter, max_pixels=MMBENCH_MAX_PIXELS)
    t0 = time.time()
    with preds_path.open("a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(todo), args.batch_size), desc="MMBench-ru"):
            batch = todo[i: i + args.batch_size]
            images = [ds[it["row"]]["image"].convert("RGB") for it in batch]
            prompts = [mc_prompt(it["question"], it["options"], it["hint"]) for it in batch]
            preds = generate_batch(model, processor, images, prompts, max_new_tokens=32)
            for it, p in zip(batch, preds):
                letter = extract_letter(p, it["options"])
                r = {k: it[k] for k in ("index", "answer", "category", "l2")}
                r.update(pred=p, letter=letter, correct=int(letter == it["answer"]))
                done[it["index"]] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

    rows = [done[it["index"]] for it in items]
    metrics = summarize(rows)
    metrics.update(adapter=args.adapter, seconds=round(time.time() - t0, 1))
    save_json(metrics, out_dir / f"mmbench_{suffix}_metrics.json")
    print(json.dumps({k: metrics[k] for k in ("n_questions", "circular_acc", "vanilla_acc", "unparsed_rate")}))


if __name__ == "__main__":
    main()
