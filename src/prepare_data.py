"""Сборка обучающей смеси из открытых данных VK (deepvk).

Источники:
  * deepvk/GQA-ru (train_balanced) — вопросы по изображениям с ответом одним словом;
  * deepvk/LLaVA-Instruct-ru — развёрнутые диалоги по изображениям COCO.

Из них собираются три вида примеров:
  1. gqa      — «вопрос + Ответь одним словом.» -> ответ (формат бенчмарка GQA-ru);
  2. gqa_mc   — тот же вопрос с 4 вариантами в формате MMBench -> буква. Дистракторы берутся
                из ответов на вопросы того же детального типа (types.detailed), например другие цвета;
  3. llava    — развёрнутые ответы (регуляризация: модель не должна разучиться отвечать подробно).
Отдельно откладывается hold-out (gqa_val.jsonl) — вопросы к картинкам, не попавшим в train.

Пример:
    python src/prepare_data.py --n-images 6000 --n-mc 2000 --n-llava 800
"""
from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image
from tqdm import tqdm

from common import DATA_DIR, OPTION_LETTERS, RAW_DIR, gqa_prompt, mc_prompt

GQA_RAW = RAW_DIR / "GQA-ru"
LLAVA_RAW = RAW_DIR / "LLaVA-Instruct-ru"
IMG_MAX_SIDE = 640  # картинки GQA ~500x375, сохраняем с запасом над TRAIN_MAX_PIXELS


def save_image(img: Image.Image, path: Path):
    img = img.convert("RGB")
    if max(img.size) > IMG_MAX_SIDE:
        img.thumbnail((IMG_MAX_SIDE, IMG_MAX_SIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92)


def mc_pools(qs):
    """Пулы дистракторов по детальному типу вопроса. Пропускаем «почти бинарные» типы (слева/справа,
    сверху/снизу): там дистракторы оказываются синонимами правильного ответа («слева» / «с левой»)."""
    raw = defaultdict(list)
    for q in qs:
        raw[q["detailed"]].append(q["answer"].strip().lower())
    pools = {}
    for t, answers in raw.items():
        cnt = Counter(answers)
        top2 = sum(n for _, n in cnt.most_common(2)) / len(answers)
        if len(cnt) >= 8 and top2 <= 0.5:
            pools[t] = answers
    return pools


def make_mc(row, pools, rng):
    """Вопрос с вариантами из GQA: правильный ответ + 3 правдоподобных дистрактора того же типа."""
    gold = row["answer"].strip().lower()
    if row["detailed"] not in pools:
        return None
    # отсекаем словоформы того же слова («темного» / «темный») — грубо, по первым 4 буквам
    pool = [a for a in pools[row["detailed"]] if a[:4] != gold[:4]]
    if len({a[:4] for a in pool}) < 3:
        return None
    # частотная выборка: дистракторы — типичные ответы этого типа вопросов
    distr = []
    while len(distr) < 3:
        a = rng.choice(pool)
        if all(a[:4] != d[:4] for d in distr):
            distr.append(a)
    opts = distr + [gold]
    rng.shuffle(opts)
    options = dict(zip(OPTION_LETTERS, opts))
    letter = OPTION_LETTERS[opts.index(gold)]
    return options, letter


def build_gqa(args, rng):
    qs = pq.read_table(GQA_RAW / "train_balanced_instructions" / "train-00000-of-00001.parquet",
                       columns=["id", "imageId", "question", "answer", "types"]).to_pylist()
    for q in qs:
        t = q.pop("types")
        q.update(structural=t["structural"], semantic=t["semantic"], detailed=t["detailed"])
    by_img = defaultdict(list)
    for q in qs:
        by_img[q["imageId"]].append(q)
    pools = mc_pools(qs)
    print(f"типов вопросов для MC-аугментации: {len(pools)}")

    img_ids = sorted(by_img)
    rng.shuffle(img_ids)
    val_imgs = set(img_ids[: args.n_val_images])
    train_imgs = set(img_ids[args.n_val_images: args.n_val_images + args.n_images])
    need = val_imgs | train_imgs
    print(f"GQA-ru train: {len(qs)} вопросов, {len(img_ids)} картинок; берём train={len(train_imgs)}, "
          f"val={len(val_imgs)}")

    # вытаскиваем только нужные картинки из parquet на 3.9 ГБ, читая его по row group
    pf = pq.ParquetFile(GQA_RAW / "train_balanced_images" / "train-00000-of-00001.parquet")
    img_dir = DATA_DIR / "gqa_images"
    saved = 0
    for rg in tqdm(range(pf.num_row_groups), desc="GQA images"):
        ids = pf.read_row_group(rg, columns=["id"]).column("id").to_pylist()
        if not need.intersection(ids):
            continue
        tbl = pf.read_row_group(rg, columns=["id", "image"]).to_pylist()
        for r in tbl:
            if r["id"] in need:
                p = img_dir / f"{r['id']}.jpg"
                if not p.exists():
                    save_image(Image.open(io.BytesIO(r["image"]["bytes"])), p)
                saved += 1
    print(f"сохранено картинок: {saved}")

    train, mc_candidates, val = [], [], []
    for iid in sorted(train_imgs):
        for q in by_img[iid]:
            img = f"gqa_images/{iid}.jpg"
            train.append(dict(source="gqa", image=img, messages=[
                {"role": "user", "content": gqa_prompt(q["question"])},
                {"role": "assistant", "content": q["answer"]}]))
            # «choose»-вопросы уже содержат 2 варианта в тексте, да/нет — тривиальны; берём открытые query
            if q["structural"] == "query":
                mc_candidates.append((img, q))
    for iid in sorted(val_imgs):
        for q in by_img[iid]:
            val.append(dict(id=q["id"], image=f"gqa_images/{iid}.jpg", question=q["question"],
                            answer=q["answer"], structural=q["structural"], semantic=q["semantic"]))

    rng.shuffle(mc_candidates)
    mc = []
    for img, q in mc_candidates:
        if len(mc) >= args.n_mc:
            break
        res = make_mc(q, pools, rng)
        if res:
            options, letter = res
            mc.append(dict(source="gqa_mc", image=img, messages=[
                {"role": "user", "content": mc_prompt(q["question"], options)},
                {"role": "assistant", "content": letter}]))
    return train, mc, val


def coco_url(path: str) -> str:
    name = Path(path).name
    split = "train2014" if "train2014" in name else ("val2014" if "val2014" in name else "train2017")
    return f"http://images.cocodataset.org/{split}/{name}"


def build_llava(args, rng):
    rows = json.loads((LLAVA_RAW / "llava_instruct_ru_train.json").read_text(encoding="utf-8"))
    rng.shuffle(rows)
    out, img_dir = [], DATA_DIR / "coco_images"
    picked = []
    for r in rows:
        conv = r["conversations"]
        # 1–2 обмена репликами и умеренная длина — чтобы последовательности влезали в 8 ГБ
        conv = conv[:4] if len(conv) >= 4 else conv[:2]
        if len(conv) < 2 or sum(len(c["value"]) for c in conv) > args.llava_max_chars:
            continue
        picked.append((r, conv))
        if len(picked) >= args.n_llava:
            break

    def fetch(item):
        r, _ = item
        p = img_dir / Path(r["image"]).name
        if p.exists():
            return True
        try:
            resp = requests.get(coco_url(r["image"]), timeout=60)
            resp.raise_for_status()
            save_image(Image.open(io.BytesIO(resp.content)), p)
            return True
        except Exception as e:  # noqa: BLE001
            print("coco fail:", r["image"], e)
            return False

    with ThreadPoolExecutor(32) as ex:
        ok = list(tqdm(ex.map(fetch, picked), total=len(picked), desc="COCO images"))
    for (r, conv), good in zip(picked, ok):
        if not good:
            continue
        msgs = []
        for c in conv:
            role = "user" if c["from"] == "human" else "assistant"
            msgs.append({"role": role, "content": c["value"].replace("<image>", "").strip()})
        out.append(dict(source="llava", image=f"coco_images/{Path(r['image']).name}", messages=msgs))
    return out


def write_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=6000, help="картинок GQA-ru для обучения")
    ap.add_argument("--n-val-images", type=int, default=300, help="картинок GQA-ru для hold-out")
    ap.add_argument("--n-mc", type=int, default=2000)
    ap.add_argument("--n-llava", type=int, default=800)
    ap.add_argument("--llava-max-chars", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    gqa, mc, val = build_gqa(args, rng)
    llava = build_llava(args, rng) if args.n_llava else []
    train = gqa + mc + llava
    rng.shuffle(train)
    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(val, DATA_DIR / "gqa_val.jsonl")
    stats = dict(total=len(train), by_source=dict(Counter(r["source"] for r in train)), val=len(val), args=vars(args))
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
