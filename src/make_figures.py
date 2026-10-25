"""Графики и сводная таблица для отчёта: results/figures/*.png, results/summary.json.

    python src/make_figures.py
"""
from __future__ import annotations

import csv
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS_DIR, save_json  # noqa: E402

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

FIG = RESULTS_DIR / "figures"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ACCENT, ACCENT2, DEEMPH = "#2a78d6", "#eb6834", "#c3c2b7"

# опубликованные метрики (карточки моделей deepvk, lmms-eval; MMBench без GPT-судьи, CircularEval)
PUBLISHED = {
    "deepvk/llava-gemma-2b-lora (3B)": {"gqa": 46.37, "mmbench": 40.19},
    "deepvk/llava-saiga-8b (8B)": {"gqa": 51.44, "mmbench": 56.65},
    "llava-hf/llava-1.5-7b-hf (7B)": {"gqa": 28.39, "mmbench": 52.25},
    "llava-v1.6-mistral-7b-hf (7B)": {"gqa": 6.65, "mmbench": 48.80},
}
OURS = {"zeroshot": "Qwen3-VL-2B, без дообучения", "lora": "Qwen3-VL-2B + LoRA (наша)"}


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def load(tag, name):
    p = RESULTS_DIR / tag / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def collect():
    rows = {k: dict(v, source="published") for k, v in PUBLISHED.items()}
    for tag, label in OURS.items():
        g, m = load(tag, "gqa_testdev_metrics.json"), load(tag, "mmbench_dev_metrics.json")
        if g or m:
            rows[label] = dict(gqa=g and g["exact_match"], mmbench=m and m["circular_acc"],
                               gqa_relaxed=g and g["relaxed_match"], mmbench_vanilla=m and m["vanilla_acc"],
                               source="ours")
    return rows


def bars(rows, key, title, path):
    items = [(k, v[key]) for k, v in rows.items() if v.get(key) is not None]
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(items) + 1.0), dpi=200)
    colors = [ACCENT if k == OURS["lora"] else (ACCENT2 if k == OURS["zeroshot"] else DEEMPH) for k, _ in items]
    y = range(len(items))
    ax.barh(y, [v for _, v in items], height=0.55, color=colors)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.8, i, f"{math.floor(v * 10 + 0.5) / 10:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), [k for k, _ in items], fontsize=9, color=INK)
    ax.set_xlim(0, max(v for _, v in items) * 1.15)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def loss_curve(path):
    log = RESULTS_DIR / "train_log.csv"
    if not log.exists():
        return
    steps, loss = [], []
    with log.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            steps.append(int(r["step"]))
            loss.append(float(r["loss"]))
    k = 30  # скользящее среднее: лосс шумит из-за разного состава батчей (GQA vs LLaVA)
    smooth = [sum(loss[max(0, i - k + 1): i + 1]) / len(loss[max(0, i - k + 1): i + 1]) for i in range(len(loss))]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    ax.plot(steps, loss, color=ACCENT, alpha=0.25, linewidth=1)
    ax.plot(steps, smooth, color=ACCENT, linewidth=2, solid_capstyle="round")
    ax.set_xlabel("шаг оптимизатора", color=INK2, fontsize=9)
    ax.set_ylabel("loss", color=INK2, fontsize=9)
    ax.set_title("Функция потерь при дообучении (скользящее среднее по 30 шагам)", loc="left", fontsize=11,
                 color=INK, pad=10)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def breakdown(path, kind="by_structural", title="GQA-ru: точность по типам вопросов"):
    a, b = load("zeroshot", "gqa_testdev_metrics.json"), load("lora", "gqa_testdev_metrics.json")
    if not (a and b):
        return
    keys = [k for k in b[kind] if k in a[kind]]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(keys) + 1.3), dpi=200)
    h = 0.36
    for i, k in enumerate(keys):
        ax.barh(i + h / 2 + 0.02, a[kind][k]["em"], height=h, color=DEEMPH, label=OURS["zeroshot"] if i == 0 else None)
        ax.barh(i - h / 2 - 0.02, b[kind][k]["em"], height=h, color=ACCENT, label=OURS["lora"] if i == 0 else None)
        ax.text(b[kind][k]["em"] + 0.8, i - h / 2 - 0.02, f"{b[kind][k]['em']:.1f}", va="center", fontsize=8,
                color=INK)
    ax.set_yticks(range(len(keys)), [f"{k} (n={b[kind][k]['n']})" for k in keys], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = collect()
    save_json(rows, RESULTS_DIR / "summary.json")
    bars(rows, "gqa", "GQA-ru (testdev, ExactMatch, %)", FIG / "gqa_ru.png")
    bars(rows, "mmbench", "MMBench-ru (dev, CircularEval, %)", FIG / "mmbench_ru.png")
    loss_curve(FIG / "train_loss.png")
    breakdown(FIG / "gqa_by_type.png")
    breakdown(FIG / "gqa_by_semantic.png", "by_semantic", "GQA-ru: точность по семантике вопроса")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
