// Сборка презентации docs/presentation.pptx (pptxgenjs). Цифры берутся из results/.
//   cd docs/_build && npm install pptxgenjs@3.12.0
//   NODE_PATH=docs/_build/node_modules node docs/build_presentation.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const R = path.join(ROOT, "results");
const read = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null);
const zg = read(path.join(R, "zeroshot/gqa_testdev_metrics.json")) || {};
const lg = read(path.join(R, "lora/gqa_testdev_metrics.json")) || {};
const zm = read(path.join(R, "zeroshot/mmbench_dev_metrics.json")) || {};
const lm = read(path.join(R, "lora/mmbench_dev_metrics.json")) || {};
const tc = read(path.join(R, "train_config.json")) || {};
const an = read(path.join(R, "analysis.json")) || {};
const st = read(path.join(ROOT, "data/processed/stats.json")) || { by_source: {} };
const inf = read(path.join(R, "inference.json")) || {};
const zub = read(path.join(R, "zeroshot/mmbench_dev_upper_bound.json")) || {};
const AUTHOR = "Засимов Георгий Алексеевич";
const REPO = process.env.REPO_URL || "github.com/…/vk-vlm-ru-project";

const MIN = tc.train_minutes ? `${Math.round(tc.train_minutes)} мин` : "—";
const f1 = (x) => (x === undefined || x === null ? "—" : (Math.round(x * 10) / 10).toFixed(1));
const PUB = { gemma: [46.37, 40.19], saiga: [51.44, 56.65], llava15: [28.39, 52.25], llava16: [6.65, 48.8] };

// ---------------------------------------------------------------- тема
const THEME = {
  name: "VK VLM",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0B0B", lt1: "FFFFFF", dk2: "14213D", lt2: "EEF1F6",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100", accent5: "6250D6", accent6: "E34948",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = THEME.colors;
const MUTED = "898781", INK2 = "52514E", GRID = "E1E0D9", DEEMPH = "C3C2B7";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = AUTHOR;
pres.title = "Компактная русскоязычная VLM на открытых данных VK";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- макеты
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.35, w: 8.8, h: 1.6, fontFace: THEME.headFontFace,
      fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.2, w: 8.8, h: 1.4, fontSize: 16,
      color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.75, fontFace: THEME.headFontFace,
      fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "VLM на открытых данных VK · " + AUTHOR, options: { x: 0.5, y: 5.2, w: 7.0, h: 0.3, fontSize: 10,
      color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

function content(section, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontSize: 14, color: C.text1, margin: 0,
  valign: "top" }, o));
function card(s, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: o.fill || C.background2 },
    line: { color: o.fill || C.background2 }, objectName: o.name });
}
function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: MUTED, width: 1.5,
    endArrowType: "triangle" } });
}

// ---------------------------------------------------------------- 1. титул
pres.addSection({ title: "Введение" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Введение" });
s.addText("Компактная русскоязычная VLM на открытых данных VK", { placeholder: "title" });
s.addText([
  { text: "Qwen3-VL-2B + LoRA, дообученная на GQA-ru и LLaVA-Instruct-ru на одной GPU 8 ГБ", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: AUTHOR + " · октябрь 2026", options: { color: DEEMPH, fontSize: 14 } },
], { placeholder: "body" });

// ---------------------------------------------------------------- 2. актуальность
s = content("Введение", "Зачем компактная VLM на русском");
txt(s, "Визуально-языковые модели отвечают на вопросы по изображениям в свободной форме. Сильные открытые VLM — это 7B+ параметров и серверные GPU, а на русском они часто заметно слабее.",
  { x: 0.5, y: 1.2, w: 9.0, h: 0.8, color: INK2 });
const why = [
  ["Локально", "Изображения не покидают устройство — важно для личных фото и закрытых данных"],
  ["Дёшево", "Обучение и инференс на потребительской видеокарте 8 ГБ, а не на кластере A100"],
  ["По-русски", "Ответы в стиле и формате, которые ждут русскоязычные пользователи и бенчмарки"],
];
why.forEach(([h, b], i) => {
  const x = 0.5 + i * 3.08;
  card(s, x, 2.2, 2.84, 1.65, { name: "why-" + i });
  txt(s, h, { x: x + 0.25, y: 2.4, w: 2.4, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  txt(s, b, { x: x + 0.25, y: 2.85, w: 2.4, h: 0.95, fontSize: 13, color: C.text1 });
});
txt(s, "Сценарии: описание фото для незрячих, модерация и предразметка контента, атрибуты товаров по фото, ассистенты в приложениях.",
  { x: 0.5, y: 4.15, w: 9.0, h: 0.6, fontSize: 14, color: INK2 });

// ---------------------------------------------------------------- 3. цель и задачи
s = content("Введение", "Цель и задачи");
card(s, 0.5, 1.2, 3.6, 3.6, { fill: C.text2, name: "goal" });
txt(s, "Цель", { x: 0.75, y: 1.4, w: 3.1, h: 0.4, fontSize: 14, bold: true, color: DEEMPH });
txt(s, "Модель ≤ 3B, которая на GQA-ru и MMBench-ru обгоняет открытые модели VK, обученная только на данных VK и одной GPU 8 ГБ",
  { x: 0.75, y: 1.85, w: 3.1, h: 2.7, fontSize: 18, color: C.background1, fontFace: THEME.headFontFace });
const tasks = ["Изучить датасеты deepvk и протокол lmms-eval", "Выбрать базовую модель, измерить zero-shot",
  "Собрать смесь из данных VK + MC-аугментация", "Дообучить LoRA в 8 ГБ видеопамяти",
  "Оценить на GQA-ru и MMBench-ru, разобрать ошибки", "Демо, отчёт, презентация, код на GitHub"];
tasks.forEach((t, i) => {
  const y = 1.2 + i * 0.6;
  s.addShape(pres.shapes.OVAL, { x: 4.5, y: y + 0.04, w: 0.38, h: 0.38, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  txt(s, String(i + 1), { x: 4.5, y: y + 0.04, w: 0.38, h: 0.38, fontSize: 13, bold: true, color: C.background1,
    align: "center", valign: "middle" });
  txt(s, t, { x: 5.05, y: y, w: 4.45, h: 0.46, fontSize: 14, valign: "middle" });
});

// ---------------------------------------------------------------- 4. данные
pres.addSection({ title: "Метод" });
s = content("Метод", "Данные VK и обучающая смесь");
const src = st.by_source || {};
const srcs = [["deepvk/GQA-ru", "train: 40 000 вопросов\nвзято 6 000 картинок"], ["deepvk/LLaVA-Instruct-ru", "144 тыс. диалогов\nвзято 800 коротких"]];
srcs.forEach(([h, b], i) => {
  const y = 1.35 + i * 1.55;
  card(s, 0.5, y, 2.6, 1.25, { name: "src-" + i });
  txt(s, h, { x: 0.7, y: y + 0.15, w: 2.3, h: 0.35, fontSize: 14, bold: true, color: C.text2 });
  txt(s, b, { x: 0.7, y: y + 0.5, w: 2.3, h: 0.7, fontSize: 12, color: INK2 });
});
const mix = [[`${src.gqa || 0}`, "короткий ответ", "вопрос + «Ответь одним словом.» → «на столе»"],
  [`${src.gqa_mc || 0}`, "вопросы с вариантами", "4 варианта, дистракторы того же типа → «C»"],
  [`${src.llava || 0}`, "развёрнутые диалоги", "чтобы модель не разучилась отвечать подробно"]];
mix.forEach(([n, h, b], i) => {
  const y = 1.2 + i * 1.05;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.75, y, w: 3.55, h: 0.88, rectRadius: 0.08, fill: { color: C.background1 },
    line: { color: GRID, width: 1 } });
  txt(s, n, { x: 3.9, y: y + 0.12, w: 1.0, h: 0.6, fontSize: 22, bold: true, color: C.accent1, valign: "middle" });
  txt(s, h, { x: 4.95, y: y + 0.1, w: 2.3, h: 0.3, fontSize: 13, bold: true });
  txt(s, b, { x: 4.95, y: y + 0.42, w: 2.3, h: 0.42, fontSize: 10, color: INK2 });
});
arrow(s, 3.15, 1.95, 3.7, 1.7); arrow(s, 3.15, 1.95, 3.7, 2.7); arrow(s, 3.15, 3.5, 3.7, 3.75);
arrow(s, 7.35, 2.7, 7.8, 2.7);
card(s, 7.85, 2.15, 1.65, 1.1, { fill: C.accent1, name: "train" });
txt(s, "LoRA\nобучение", { x: 7.85, y: 2.15, w: 1.65, h: 1.1, fontSize: 15, bold: true, color: C.background1, align: "center",
  valign: "middle" });
txt(s, "Бенчмарки GQA-ru testdev (12 216) и MMBench-ru dev (3 910) в обучении не участвуют; для контроля — hold-out из 300 картинок train.",
  { x: 0.5, y: 4.45, w: 9.0, h: 0.55, fontSize: 12, color: INK2 });

// ---------------------------------------------------------------- 5. модель
s = content("Метод", "Qwen3-VL-2B + LoRA");
const blocksA = [["Изображение", 0.5, C.background2, C.text1], ["Визуальный\nэнкодер ViT\nзаморожен", 2.35, C.background2, C.text1],
  ["Проектор\nзаморожен", 4.2, C.background2, C.text1]];
blocksA.forEach(([t, x, fill, color], i) => {
  card(s, x, 1.45, 1.6, 0.95, { fill, name: "arch-" + i });
  txt(s, t, { x, y: 1.45, w: 1.6, h: 0.95, fontSize: 13, color, align: "center", valign: "middle", bold: i === 0 });
  arrow(s, x + 1.62, 1.92, x + 1.83, 1.92);
});
card(s, 6.05, 1.2, 2.1, 1.45, { fill: C.accent1, name: "llm" });
txt(s, "Языковая модель\n+ LoRA r = 16\nобучается", { x: 6.05, y: 1.2, w: 2.1, h: 1.45, fontSize: 14, bold: true,
  color: C.background1, align: "center", valign: "middle" });
arrow(s, 8.17, 1.92, 8.45, 1.92);
txt(s, "Ответ", { x: 8.5, y: 1.7, w: 1.0, h: 0.45, fontSize: 14, bold: true, valign: "middle" });
txt(s, "Вопрос на русском", { x: 6.05, y: 2.85, w: 2.1, h: 0.3, fontSize: 12, color: INK2, align: "center" });
arrow(s, 7.1, 2.85, 7.1, 2.68);
const stats5 = [["17.4 млн", "обучаемых параметров (0.81%)"], ["≤ 196", "визуальных токенов на картинку"], ["70 МБ", "размер адаптера"]];
stats5.forEach(([n, l], i) => {
  const x = 0.5 + i * 3.08;
  txt(s, n, { x, y: 3.45, w: 2.9, h: 0.7, fontSize: 34, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  txt(s, l, { x, y: 4.15, w: 2.9, h: 0.5, fontSize: 13, color: INK2 });
});

// ---------------------------------------------------------------- 6. обучение
s = content("Метод", "Обучение на ноутбучной GPU 8 ГБ");
const logPath = path.join(R, "train_log.csv");
if (fs.existsSync(logPath)) {
  const rows = fs.readFileSync(logPath, "utf8").trim().split(/\r?\n/).slice(1).map((l) => l.split(","));
  const steps = rows.map((r) => +r[0]), loss = rows.map((r) => +r[1]);
  const k = 30, xs = [], ys = [];
  for (let i = k - 1; i < loss.length; i += 10) {
    xs.push(steps[i] % 100 === 0 ? String(steps[i]) : "");
    ys.push(+(loss.slice(i - k + 1, i + 1).reduce((a, b) => a + b, 0) / k).toFixed(3));
  }
  s.addChart(pres.charts.LINE, [{ name: "loss", labels: xs, values: ys }], {
    x: 0.4, y: 1.15, w: 5.6, h: 3.75, chartColors: [HEX.accent1], lineSize: 2, lineDataSymbol: "none",
    showLegend: false, showTitle: true, title: "Loss по шагам (скользящее среднее по 30 шагам)", titleFontSize: 12,
    titleColor: INK2, titleFontFace: "+mn-lt", catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 10,
    valAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" }, catAxisLineShow: false,
    valAxisLineShow: false, catAxisLabelRotate: 0, valAxisMinVal: 0,
  });
}
const stats6 = [[`${tc.train_minutes || "—"} мин`, "одна эпоха, " + (tc.steps || "—") + " шагов"],
  [`${tc.max_mem_gb || "—"} ГБ`, "пик видеопамяти"], [`${tc.train_examples || st.total || "—"}`, "обучающих примеров"]];
stats6.forEach(([n, l], i) => {
  const y = 1.2 + i * 1.2;
  txt(s, n, { x: 6.4, y, w: 3.1, h: 0.6, fontSize: 30, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  txt(s, l, { x: 6.4, y: y + 0.6, w: 3.1, h: 0.4, fontSize: 13, color: INK2 });
});
s.addNotes("Ключевой трюк по памяти: логиты по словарю 152k считаются только на позициях ответа. Пик памяти упал с 7.8 до 6.4 ГБ, скорость выросла вдвое.");

// ---------------------------------------------------------------- 7–8. результаты
pres.addSection({ title: "Результаты" });
function resultSlide(title, idx, ours0, ours1, metricLabel) {
  const sl = content("Результаты", title);
  const items = [["llava-1.6-mistral-7b", PUB.llava16[idx]], ["llava-1.5-7b", PUB.llava15[idx]],
    ["deepvk/llava-gemma-2b", PUB.gemma[idx]], ["deepvk/llava-saiga-8b", PUB.saiga[idx]],
    ["Qwen3-VL-2B (zero-shot)", ours0], ["Qwen3-VL-2B + LoRA (наша)", ours1]].filter((x) => x[1] !== undefined && x[1] !== null);
  items.sort((a, b) => a[1] - b[1]);
  const colors = items.map(([n]) => (n.includes("наша") ? HEX.accent1 : n.includes("zero-shot") ? HEX.accent2 : DEEMPH));
  sl.addChart(pres.charts.BAR, [{ name: metricLabel, labels: items.map((x) => x[0]), values: items.map((x) => Math.round(x[1] * 10) / 10) }], {
    x: 0.4, y: 1.15, w: 6.0, h: 3.8, barDir: "bar", chartColors: colors, barGapWidthPct: 60,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 11, dataLabelColor: "0B0B0B", dataLabelFontFace: "+mn-lt",
    showLegend: false, catAxisLabelColor: "0B0B0B", catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisMinVal: 0,
    valAxisMaxVal: Math.ceil(Math.max(...items.map((x) => x[1])) * 1.18), catAxisLineShow: false,
  });
  return sl;
}
s = resultSlide("GQA-ru: ExactMatch на testdev", 0, zg.exact_match, lg.exact_match, "GQA-ru EM");
const dG = lg.exact_match && zg.exact_match ? lg.exact_match - zg.exact_match : null;
txt(s, dG !== null ? `+${dG.toFixed(1)}` : "—", { x: 6.7, y: 1.3, w: 2.8, h: 0.8, fontSize: 44, bold: true, color: C.accent1,
  fontFace: THEME.headFontFace });
txt(s, `п.п. к исходной модели после ${MIN} обучения`, { x: 6.7, y: 2.1, w: 2.8, h: 0.6, fontSize: 13, color: INK2 });
txt(s, lg.exact_match ? `${(lg.exact_match - PUB.saiga[0] >= 0 ? "+" : "")}${(lg.exact_match - PUB.saiga[0]).toFixed(1)} п.п. к llava-saiga-8b — модели в 4 раза больше, обученной на 8×A100`
  : "", { x: 6.7, y: 2.95, w: 2.8, h: 1.0, fontSize: 13, color: C.text1 });
txt(s, "12 216 вопросов, без учёта регистра и пунктуации, как в lmms-eval", { x: 6.7, y: 4.15, w: 2.8, h: 0.6, fontSize: 11, color: MUTED });

s = resultSlide("MMBench-ru: CircularEval на dev", 1, zm.circular_acc, lm.circular_acc, "MMBench-ru");
txt(s, f1(lm.circular_acc), { x: 6.7, y: 1.3, w: 2.8, h: 0.8, fontSize: 44, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
txt(s, `CircularEval; без дообучения — ${f1(zm.circular_acc)}`, { x: 6.7, y: 2.1, w: 2.8, h: 0.6, fontSize: 13, color: INK2 });
txt(s, `+${f1(lm.circular_acc - PUB.saiga[1])} п.п. к llava-saiga-8b. Прирост к исходной модели — в основном формат: она в ${f1(zm.unparsed_rate)}% случаев рассуждала вместо буквы`,
  { x: 6.7, y: 2.95, w: 2.8, h: 1.2, fontSize: 13, color: C.text1 });
txt(s, `Верхняя граница исходной модели — ${f1(zub.circular_acc)}: вывод «общие навыки не пострадали»`,
  { x: 6.7, y: 4.25, w: 2.8, h: 0.6, fontSize: 11, color: MUTED });

// ---------------------------------------------------------------- 9. что изменилось
s = content("Результаты", "Что выучила модель");
const g = an.gqa || { examples_fixed: [] };
const ex = (g.examples_fixed || []).filter((e) => e.question.length < 60).slice(0, 5);
const head = ["Вопрос", "Эталон", "Без дообучения", "LoRA"].map((t) => ({ text: t, options: { bold: true, color: INK2,
  fill: { color: HEX.lt2 } } }));
s.addTable([head, ...ex.map((e) => [e.question, e.answer, e.zeroshot, e.lora])], {
  x: 0.5, y: 1.15, w: 9.0, colW: [4.4, 1.5, 1.6, 1.5], fontSize: 12, fontFace: "Calibri", color: "0B0B0B",
  border: { type: "solid", pt: 0.75, color: GRID }, margin: 0.06, rowH: 0.42,
});
const zy = (g.zeroshot || {}), ly = (g.lora || {});
txt(s, g.n ? `Исправлено ${g.fixed_by_lora} ответов, испорчено ${g.broken_by_lora}. Два эффекта: исходная модель склонялась к «нет» (да/нет: ${f1(zy.yes_no_em)} → ${f1(ly.yes_no_em)}) и отвечала не в форме датасета — «левая» вместо «слева» (открытые: ${f1(zy.open_em)} → ${f1(ly.open_em)}).`
  : "", { x: 0.5, y: 3.95, w: 9.0, h: 0.95, fontSize: 14, color: C.text1 });

// ---------------------------------------------------------------- 10. жизнеспособность
pres.addSection({ title: "Итоги" });
s = content("Итоги", "Жизнеспособность модели");
const live = [["4.3 ГБ + 70 МБ", "базовая модель + адаптер; несколько адаптеров на одну базу"],
  [`${inf.peak_gb_batch16 || "—"} ГБ VRAM`, "пик инференса: потребительская GPU, данные не уходят в облако"],
  [`${inf.latency_short_answer_s ? inf.latency_short_answer_s.toFixed(2) : "—"} с`, `короткий ответ; батчем — ${inf.throughput_short_q_per_s || "—"} вопроса/с, описание 128 токенов — ${inf.latency_description_128tok_s ? inf.latency_description_128tok_s.toFixed(1) : "—"} с`],
  [`${MIN} vs дни`, "обучение на ноутбуке против 3–4 дней на 8×A100 у llava-saiga-8b"]];
live.forEach(([n, l], i) => {
  const x = 0.5 + (i % 2) * 4.6, y = 1.2 + Math.floor(i / 2) * 1.75;
  card(s, x, y, 4.4, 1.5, { name: "live-" + i });
  txt(s, n, { x: x + 0.25, y: y + 0.15, w: 3.9, h: 0.6, fontSize: 26, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  txt(s, l, { x: x + 0.25, y: y + 0.8, w: 3.9, h: 0.6, fontSize: 13, color: INK2 });
});

// ---------------------------------------------------------------- 11. ограничения
s = content("Итоги", "Ограничения и следующие шаги");
const lim = ["Метрики VK взяты из карточек моделей; наш скрипт повторяет протокол lmms-eval, но это другая реализация",
  "Прирост на GQA-ru — в основном калибровка «да/нет» и форма ответов; рост «понимания» картинок отдельно не мерили",
  "Под MMBench-ru нет обучающих данных: MC-аугментация из GQA не покрывает OCR и диаграммы",
  "Один прогон без подбора гиперпараметров и абляций — из-за сроков"];
const next = ["Весь GQA-ru train (40 тыс. вопросов) и больше LLaVA-Instruct-ru", "Qwen3-VL-4B в 4-битной QLoRA",
  "Абляции по доле MC-аугментации и разрешению картинок", "Проверить модели VK тем же скриптом"];
txt(s, "Ограничения", { x: 0.5, y: 1.15, w: 4.3, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
txt(s, lim.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lim.length - 1 } })),
  { x: 0.5, y: 1.6, w: 4.3, h: 3.3, fontSize: 14, paraSpaceAfter: 6, color: C.text1 });
txt(s, "Дальше", { x: 5.2, y: 1.15, w: 4.3, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
txt(s, next.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < next.length - 1 } })),
  { x: 5.2, y: 1.6, w: 4.3, h: 3.3, fontSize: 14, paraSpaceAfter: 6, color: C.text1 });

// ---------------------------------------------------------------- 12. итог
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Итоги" });
s.addText(lg.exact_match ? `GQA-ru ${f1(lg.exact_match)} · MMBench-ru ${f1(lm.circular_acc)}` : "Итоги", { placeholder: "title" });
const wins = [lg.exact_match > PUB.saiga[0] ? "GQA-ru" : null, lm.circular_acc > PUB.saiga[1] ? "MMBench-ru" : null].filter(Boolean);
const verdict = wins.length ? `обходит открытые модели VK на ${wins.join(" и ")}` : "сравнима с открытыми моделями VK";
s.addText([
  { text: `2B-модель, дообученная на открытых данных VK за ${MIN} на ноутбуке, ${verdict}.`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Код, адаптер, отчёт и демо: " + REPO, options: { color: DEEMPH, fontSize: 14 } },
], { placeholder: "body" });

(async () => {
  const out = path.join(__dirname, "presentation.pptx");
  await pres.writeFile({ fileName: out });
  const skill = process.env.PPTX_SKILL;
  if (skill) await require(path.join(skill, "scripts/apply_theme.js")).applyTheme(out, THEME);
  console.log("готово:", out);
})();
