"""Проверка чувствительности MMBench-ru к лимиту генерации.

В основном прогоне ответ ограничен 32 токенами. Если модель начинает рассуждать («Рассмотрим…»), букву
извлечь не удаётся. lmms-eval даёт до 1024 токенов, поэтому перегенерируем только нераспознанные записи
с длинным лимитом и пересчитываем метрики -> results/<tag>/mmbench_dev_long_metrics.json.

    python src/mmbench_retry.py --tag zeroshot --max-new-tokens 512
"""
from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from common import MMBENCH_MAX_PIXELS, RESULTS_DIR, extract_letter, generate_batch, load_model_and_processor, mc_prompt, \
    save_json
from eval_mmbench import load_mmbench, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    out_dir = RESULTS_DIR / args.tag
    rows = {r["index"]: r for r in map(json.loads, (out_dir / "mmbench_dev_preds.jsonl").open(encoding="utf-8"))}
    items, ds = load_mmbench()
    todo = [it for it in items if rows[it["index"]]["letter"] is None]
    print(f"нераспознанных записей: {len(todo)}")

    model, processor = load_model_and_processor(args.adapter, max_pixels=MMBENCH_MAX_PIXELS)
    recovered = 0
    for i in tqdm(range(0, len(todo), args.batch_size), desc="retry"):
        batch = todo[i: i + args.batch_size]
        preds = generate_batch(model, processor, [ds[it["row"]]["image"].convert("RGB") for it in batch],
                               [mc_prompt(it["question"], it["options"], it["hint"]) for it in batch],
                               max_new_tokens=args.max_new_tokens)
        for it, p in zip(batch, preds):
            letter = extract_letter(p, it["options"])
            recovered += letter is not None
            rows[it["index"]].update(pred=p, letter=letter, correct=int(letter == it["answer"]))

    with (out_dir / "mmbench_dev_long_preds.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(rows[it["index"]], ensure_ascii=False) + "\n")
    metrics = summarize([rows[it["index"]] for it in items])
    metrics.update(retried=len(todo), recovered=recovered, max_new_tokens=args.max_new_tokens)
    save_json(metrics, out_dir / "mmbench_dev_long_metrics.json")
    print(json.dumps({k: metrics[k] for k in ("retried", "recovered", "circular_acc", "vanilla_acc", "unparsed_rate")}))


if __name__ == "__main__":
    main()
