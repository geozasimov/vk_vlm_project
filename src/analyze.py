"""Анализ ошибок: сравнение zero-shot и LoRA по предсказаниям -> results/analysis.json и results/analysis.md.

    python src/analyze.py
"""
from __future__ import annotations

import json
import random
from collections import Counter

from common import RESULTS_DIR, norm_exact, save_json


def read(path):
    return {r.get("id", r.get("index")): r for r in map(json.loads, path.open(encoding="utf-8"))}


def gqa_stats(rows):
    preds = [norm_exact(r["pred"]) for r in rows.values()]
    yn = [r for r in rows.values() if r["answer"] in ("да", "нет")]
    return {
        "yes_no_pred": dict(Counter(norm_exact(r["pred"]) for r in yn if norm_exact(r["pred"]) in ("да", "нет"))),
        "yes_no_gold": dict(Counter(r["answer"] for r in yn)),
        "yes_no_em": round(100 * sum(r["em"] for r in yn) / len(yn), 2),
        "open_em": round(100 * sum(r["em"] for r in rows.values() if r["answer"] not in ("да", "нет")) /
                         sum(r["answer"] not in ("да", "нет") for r in rows.values()), 2),
        "avg_words_in_pred": round(sum(len(p.split()) for p in preds) / len(preds), 2),
        "pred_longer_than_2_words_%": round(100 * sum(len(p.split()) > 2 for p in preds) / len(preds), 2),
        "top_errors": [f"{g} → {p} ({n})" for (g, p), n in Counter(
            (norm_exact(r["answer"]), norm_exact(r["pred"])) for r in rows.values() if not r["em"]).most_common(15)],
    }


def main():
    zs_p, lo_p = RESULTS_DIR / "zeroshot" / "gqa_testdev_preds.jsonl", RESULTS_DIR / "lora" / "gqa_testdev_preds.jsonl"
    zs, lo = read(zs_p), read(lo_p)
    ids = [i for i in zs if i in lo]
    fixed = [i for i in ids if not zs[i]["em"] and lo[i]["em"]]
    broken = [i for i in ids if zs[i]["em"] and not lo[i]["em"]]
    rng = random.Random(0)

    def ex(i):
        return dict(question=zs[i]["question"], answer=zs[i]["answer"], zeroshot=zs[i]["pred"], lora=lo[i]["pred"])

    out = {
        "gqa": {
            "n": len(ids), "fixed_by_lora": len(fixed), "broken_by_lora": len(broken),
            "zeroshot": gqa_stats(zs), "lora": gqa_stats(lo),
            "examples_fixed": [ex(i) for i in rng.sample(fixed, min(12, len(fixed)))],
            "examples_broken": [ex(i) for i in rng.sample(broken, min(8, len(broken)))],
        }
    }
    mz, ml = RESULTS_DIR / "zeroshot" / "mmbench_dev_metrics.json", RESULTS_DIR / "lora" / "mmbench_dev_metrics.json"
    if mz.exists() and ml.exists():
        a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (mz, ml))
        out["mmbench"] = {
            "circular": [a["circular_acc"], b["circular_acc"]], "vanilla": [a["vanilla_acc"], b["vanilla_acc"]],
            "unparsed": [a["unparsed_rate"], b["unparsed_rate"]],
            "by_l2": {k: [a["circular_by_l2"][k]["acc"], v["acc"], v["n"]] for k, v in b["circular_by_l2"].items()},
        }
    save_json(out, RESULTS_DIR / "analysis.json")

    md = ["# Анализ ошибок", "", f"GQA-ru: LoRA исправила {len(fixed)} ответов и ухудшила {len(broken)} "
          f"(из {len(ids)}).", "", "| вопрос | эталон | zero-shot | LoRA |", "|---|---|---|---|"]
    md += [f"| {e['question']} | {e['answer']} | {e['zeroshot']} | {e['lora']} |" for e in out["gqa"]["examples_fixed"]]
    md += ["", "Ухудшения:", "", "| вопрос | эталон | zero-shot | LoRA |", "|---|---|---|---|"]
    md += [f"| {e['question']} | {e['answer']} | {e['zeroshot']} | {e['lora']} |" for e in out["gqa"]["examples_broken"]]
    (RESULTS_DIR / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out["gqa"].items() if not k.startswith("examples")}, ensure_ascii=False,
                     indent=2))
    if "mmbench" in out:
        print(json.dumps(out["mmbench"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
