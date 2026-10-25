"""LoRA-дообучение Qwen3-VL-2B-Instruct на смеси данных VK (data/processed/train.jsonl).

Замораживаем всё, кроме LoRA-адаптеров в линейных слоях языковой модели (attention + MLP).
Визуальный энкодер и проектор (merger) не трогаем: их предобучение на порядки больше наших данных,
а LLM-часть — это то, что нужно «научить» формату и русскоязычным ответам.
Лосс считается только по токенам ответов ассистента.

Пример:
    python src/train_lora.py --epochs 1 --batch-size 4 --grad-accum 4 --lr 2e-4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

from common import ADAPTER_DIR, BASE_MODEL, DATA_DIR, MIN_PIXELS, OUT_DIR, RESULTS_DIR, TRAIN_MAX_PIXELS, save_json

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def to_chat(row, image):
    msgs = []
    for j, m in enumerate(row["messages"]):
        content = [{"type": "text", "text": m["content"]}]
        if j == 0:
            content.insert(0, {"type": "image", "image": image})
        msgs.append({"role": m["role"], "content": content})
    return msgs


class Collator:
    """Токенизирует диалоги и строит labels: -100 везде, кроме ответов ассистента (включая <|im_end|>)."""

    def __init__(self, processor):
        self.p = processor
        tok = processor.tokenizer
        self.start = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.end = tok.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, rows):
        images = [Image.open(DATA_DIR / r["image"]).convert("RGB") for r in rows]
        texts = [self.p.apply_chat_template(to_chat(r, img), tokenize=False) for r, img in zip(rows, images)]
        batch = self.p(text=texts, images=images, padding=True, return_tensors="pt")
        ids = batch["input_ids"]
        labels = torch.full_like(ids, -100)
        n = len(self.start)
        for b, seq in enumerate(ids.tolist()):
            i = 0
            while i <= len(seq) - n:
                if seq[i: i + n] == self.start:
                    j = i + n
                    while j < len(seq) and seq[j] != self.end:
                        j += 1
                    labels[b, i + n: j + 1] = ids[b, i + n: j + 1]
                    i = j
                i += 1
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DATA_DIR / "train.jsonl"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None, help="оборвать после N шагов оптимизатора (отладка)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out", default=str(ADAPTER_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(line) for line in open(args.train, encoding="utf-8")]
    processor = AutoProcessor.from_pretrained(BASE_MODEL, min_pixels=MIN_PIXELS, max_pixels=TRAIN_MAX_PIXELS)
    processor.tokenizer.padding_side = "right"
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=Collator(processor),
                        num_workers=0, drop_last=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_cfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, target_modules=LORA_TARGETS,
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
    total_steps = math.ceil(len(loader) * args.epochs / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(args.warmup * total_steps)), total_steps)
    print(f"примеров: {len(rows)}, микробатчей/эпоха: {len(loader)}, шагов оптимизатора: {total_steps}")

    log_path = RESULTS_DIR / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["step", "loss", "lr", "elapsed_s", "samples_per_s", "max_mem_gb"])

    model.train()
    step, micro, run_loss, t0 = 0, 0, 0.0, time.time()
    done = False
    while not done:
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # Логиты по словарю в 152k токенов для всей последовательности не влезают в 8 ГБ, а нужны они
            # только там, где следующий токен — часть ответа. Считаем lm_head лишь на этих позициях.
            target = batch.pop("labels")[:, 1:]
            keep = (target != -100).any(0).nonzero().squeeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch, logits_to_keep=keep).logits
            loss = F.cross_entropy(logits.float().flatten(0, 1), target[:, keep].flatten(), ignore_index=-100)
            loss = loss / args.grad_accum
            loss.backward()
            run_loss += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            el = time.time() - t0
            sps = step * args.grad_accum * args.batch_size / el
            log.writerow([step, round(run_loss, 4), f"{sched.get_last_lr()[0]:.2e}", round(el, 1), round(sps, 2),
                          round(torch.cuda.max_memory_allocated() / 2**30, 2)])
            log_f.flush()
            if step % 10 == 0 or step == 1:
                eta = (total_steps - step) * el / step / 60
                print(f"step {step}/{total_steps} loss {run_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{sps:.2f} samples/s mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
                      f"ETA {eta:.0f} min", flush=True)
            run_loss = 0.0
            if step % args.save_every == 0:
                model.save_pretrained(OUT_DIR / "checkpoints" / f"step_{step}")
            if step >= total_steps:
                done = True
                break

    model.save_pretrained(args.out)
    save_json(dict(vars(args), base_model=BASE_MODEL, train_examples=len(rows), steps=step,
                   train_minutes=round((time.time() - t0) / 60, 1),
                   max_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2)),
              RESULTS_DIR / "train_config.json")
    log_f.close()
    print("адаптер сохранён:", args.out)


if __name__ == "__main__":
    main()
