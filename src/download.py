"""Многопоточная загрузка файлов с Hugging Face Hub через HTTP Range.

Хаб отдаёт ~0.4 МБ/с на одно соединение, поэтому файл режется на куски и качается в N потоков.
Пример:
    python src/download.py model          # Qwen3-VL-2B-Instruct -> models/
    python src/download.py gqa-train      # картинки GQA-ru train (3.9 ГБ) -> data/raw/
    python src/download.py llava          # LLaVA-Instruct-ru (json) -> data/raw/
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from common import ROOT

CHUNK = 16 * 1024 * 1024
WORKERS = 32

JOBS = {
    "model": [("Qwen/Qwen3-VL-2B-Instruct", "model", f, ROOT / "models" / "Qwen3-VL-2B-Instruct" / f) for f in [
        "config.json", "generation_config.json", "chat_template.json", "merges.txt", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
        "model.safetensors"]],
    "gqa-train": [("deepvk/GQA-ru", "dataset", f, ROOT / "data" / "raw" / "GQA-ru" / f) for f in [
        "train_balanced_instructions/train-00000-of-00001.parquet",
        "train_balanced_images/train-00000-of-00001.parquet"]],
    "llava": [("deepvk/LLaVA-Instruct-ru", "dataset", f, ROOT / "data" / "raw" / "LLaVA-Instruct-ru" / f)
              for f in ["llava_instruct_ru_train.json"]],
}


def hub_url(repo: str, rtype: str, fname: str) -> str:
    prefix = "datasets/" if rtype == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo}/resolve/main/{fname}"


def resolve(url: str) -> tuple[str, int]:
    """Итоговый URL после редиректа на CDN и размер файла."""
    r = requests.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, stream=True, timeout=60)
    r.raise_for_status()
    size = int(r.headers["Content-Range"].split("/")[-1]) if "Content-Range" in r.headers else int(
        r.headers.get("Content-Length", 0))
    final = r.url
    r.close()
    return final, size


def fetch_range(url: str, path: Path, start: int, end: int, progress: list, lock: threading.Lock):
    for attempt in range(8):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            data = r.content
            assert len(data) == end - start + 1, f"short read {len(data)} != {end - start + 1}"
            with lock:
                with path.open("r+b") as f:
                    f.seek(start)
                    f.write(data)
                progress[0] += len(data)
            return
        except Exception as e:  # noqa: BLE001 — повторяем любую сетевую ошибку
            time.sleep(2 + 3 * attempt)
            err = e
    raise RuntimeError(f"range {start}-{end} failed: {err}")


def download(repo: str, rtype: str, fname: str, dest: Path):
    url, size = resolve(hub_url(repo, rtype, fname))
    if dest.exists() and dest.stat().st_size == size:
        print(f"skip {dest.name} ({size / 1e6:.1f} MB) — уже есть")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        f.truncate(size)
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)] or [(0, 0)]
    progress, lock, t0 = [0], threading.Lock(), time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch_range, url, tmp, s, e, progress, lock) for s, e in ranges]
        last = 0.0
        for fut in as_completed(futs):
            fut.result()
            if time.time() - last > 10:
                last = time.time()
                mb = progress[0] / 1e6
                print(f"  {dest.name}: {mb:8.1f}/{size / 1e6:.1f} MB  {mb / (time.time() - t0):5.1f} MB/s", flush=True)
    tmp.replace(dest)
    print(f"done {dest.name} ({size / 1e6:.1f} MB) за {time.time() - t0:.0f} c", flush=True)


if __name__ == "__main__":
    for job in sys.argv[1:]:
        for args in JOBS[job]:
            download(*args)
    print("ALL DONE", flush=True)
