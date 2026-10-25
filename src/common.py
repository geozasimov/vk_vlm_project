"""Общие константы и утилиты: пути, загрузка модели, промпты, нормализация ответов."""
from __future__ import annotations

import json
import re
import string
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"

RAW_DIR = ROOT / "data" / "raw"
BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
# локальная копия (src/download.py model), иначе — загрузка с Hub
_LOCAL_MODEL = ROOT / "models" / "Qwen3-VL-2B-Instruct"
BASE_MODEL = str(_LOCAL_MODEL) if (_LOCAL_MODEL / "model.safetensors").exists() else BASE_MODEL_ID
ADAPTER_DIR = OUT_DIR / "lora-adapter"

# Разрешение картинок: Qwen3-VL режет изображение на патчи 16px и сливает 2x2,
# т.е. 1 визуальный токен = 32x32 px. 448*448 px ~ 196 токенов на картинку GQA.
TRAIN_MAX_PIXELS = 448 * 448
GQA_MAX_PIXELS = 448 * 448
MMBENCH_MAX_PIXELS = 640 * 640
MIN_PIXELS = 256 * 256  # штатное значение preprocessor_config.json модели

# Промпты в точности как в lmms-eval (задачи gqa-ru и mmbench_ru_dev),
# чтобы наши цифры были сопоставимы с опубликованными метриками моделей VK.
GQA_POST_PROMPT = "\nОтветь одним словом."
MMBENCH_POST_PROMPT = "\nВыбери правильный вариант ответа буквой."
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def gqa_prompt(question: str) -> str:
    return f"{question}{GQA_POST_PROMPT}"


def mc_prompt(question: str, options: dict[str, str], hint: str | None = None) -> str:
    """Формат MMBench из lmms-eval: '{hint} {question} Options:\\nA. ...'."""
    opts = "Options:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
    head = f"{hint} {question}" if hint else question
    return f"{head} {opts}{MMBENCH_POST_PROMPT}"


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or str(x).strip().lower() in {"", "nan", "none"}


# ---------------------------------------------------------------- нормализация
_PUNCT_RE = re.compile("[" + re.escape(string.punctuation + "«»—–…“”„") + "]")


def norm_exact(s: str) -> str:
    """Как exact_match в lmms-eval: ignore_case + ignore_punctuation."""
    return _PUNCT_RE.sub("", s.lower()).strip()


def norm_relaxed(s: str) -> str:
    """Мягкая нормализация (вторичная метрика): ё→е, цифры→слова, первое слово."""
    s = norm_exact(s).replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    try:
        from num2words import num2words

        s = re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="ru"), s)
    except ImportError:
        pass
    return s


_CYR2LAT = str.maketrans({"А": "A", "В": "B", "С": "C", "Д": "D", "Е": "E"})


def extract_letter(pred: str, options: dict[str, str]) -> str | None:
    """Достаёт букву варианта из ответа модели (аналог can_infer из MMBench)."""
    p = pred.strip()
    letters = [k for k in options]
    # 1) ответ начинается с буквы: "B", "B.", "B) ...", "(B)"
    m = re.match(r"^\(?([A-E])(?:[\s.):,*]|$)", p)
    if m and m.group(1) in letters:
        return m.group(1)
    # 1б) кириллические двойники («А.», «В)») — только с точкой/скобкой: «В комнате…» — это предлог
    m = re.match(r"^\(?([АВСДЕ])(?:[.):]|$)", p)
    if m and m.group(1).translate(_CYR2LAT) in letters:
        return m.group(1).translate(_CYR2LAT)
    # 2) ровно одна отдельно стоящая латинская буква среди допустимых. Соседей проверяем по любым буквам,
    # включая кириллицу, чтобы не ловить буквы внутри слов
    found = {c for c in re.findall(r"(?<![^\W\d_])([A-E])(?![^\W\d_])", p) if c in letters}
    if len(found) == 1:
        return found.pop()
    # 3) совпадение по тексту варианта
    pl = norm_exact(pred)
    hits = [k for k, v in options.items() if norm_exact(str(v)) and norm_exact(str(v)) in pl]
    if len(hits) == 1:
        return hits[0]
    return None


# ---------------------------------------------------------------- модель
def load_model_and_processor(adapter: str | Path | None = None, max_pixels: int = GQA_MAX_PIXELS,
                             padding_side: str = "left"):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(BASE_MODEL, min_pixels=MIN_PIXELS, max_pixels=max_pixels)
    processor.tokenizer.padding_side = padding_side
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
    model.eval()
    return model, processor


def build_messages(image, text: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]


@torch.inference_mode()
def generate_batch(model, processor, images, prompts, max_new_tokens: int = 16) -> list[str]:
    texts = [
        processor.apply_chat_template(build_messages(img, p), tokenize=False, add_generation_prompt=True)
        for img, p in zip(images, prompts)
    ]
    inputs = processor(text=texts, images=list(images), padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=processor.tokenizer.pad_token_id)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return [s.strip() for s in processor.batch_decode(gen, skip_special_tokens=True)]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
