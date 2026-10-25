"""Gradio-демо: загрузите картинку и задайте вопрос по-русски.

    python src/demo.py                       # дообученная модель (outputs/lora-adapter)
    python src/demo.py --adapter none        # исходная Qwen3-VL-2B-Instruct для сравнения
"""
from __future__ import annotations

import argparse

import gradio as gr
import torch

from common import ADAPTER_DIR, GQA_POST_PROMPT, MMBENCH_MAX_PIXELS, build_messages, load_model_and_processor

MODES = {
    "Подробно": ("", 384),
    "Одним словом": (GQA_POST_PROMPT, 16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ADAPTER_DIR))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    adapter = None if args.adapter == "none" else args.adapter
    model, processor = load_model_and_processor(adapter, max_pixels=MMBENCH_MAX_PIXELS)

    @torch.inference_mode()
    def answer(image, question, mode):
        if image is None or not question.strip():
            return "Загрузите изображение и введите вопрос."
        suffix, max_new = MODES[mode]
        text = processor.apply_chat_template(build_messages(image, question.strip() + suffix), tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05)
        return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    title = "Qwen3-VL-2B + LoRA на данных VK" if adapter else "Qwen3-VL-2B-Instruct (без дообучения)"
    demo = gr.Interface(
        fn=answer,
        inputs=[gr.Image(type="pil", label="Изображение"),
                gr.Textbox(label="Вопрос", placeholder="Что изображено на картинке?"),
                gr.Radio(list(MODES), value="Подробно", label="Формат ответа")],
        outputs=gr.Textbox(label="Ответ модели"),
        title=title,
        description="Русскоязычная визуально-языковая модель, дообученная на открытых датасетах deepvk "
                    "(GQA-ru, LLaVA-Instruct-ru). Работает локально на GPU 8 ГБ.",
        flagging_mode="never",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
