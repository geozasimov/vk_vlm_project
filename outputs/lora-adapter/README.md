---
base_model: Qwen/Qwen3-VL-2B-Instruct
library_name: peft
license: apache-2.0
language:
- ru
tags:
- lora
- vision-language
- visual-question-answering
datasets:
- deepvk/GQA-ru
- deepvk/LLaVA-Instruct-ru
---

# Qwen3-VL-2B-Instruct + LoRA на данных VK

LoRA-адаптер для [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), дообученный на
открытых русскоязычных данных VK: [deepvk/GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru) (train) и
[deepvk/LLaVA-Instruct-ru](https://huggingface.co/datasets/deepvk/LLaVA-Instruct-ru). Метрики, протокол оценки и
подробности — в README и отчёте репозитория.

* LoRA r=16, α=32, dropout 0.05, все линейные слои языковой модели (17.4 млн параметров); визуальная часть заморожена.
* 11 530 примеров, 1 эпоха, ~38 минут на RTX 5060 Laptop 8 GB.
* Короткий ответ: добавьте к вопросу `\nОтветь одним словом.`; вопрос с вариантами — формат MMBench
  (`Options:\nA. ...` + `\nВыбери правильный вариант ответа буквой.`).

```python
import torch
from transformers import Qwen3VLForConditionalGeneration
from peft import PeftModel

model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16,
                                                        device_map="cuda")
model = PeftModel.from_pretrained(model, "outputs/lora-adapter").merge_and_unload()
```
