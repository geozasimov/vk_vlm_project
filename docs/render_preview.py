"""Превью документа для визуальной проверки: .pptx/.docx -> PDF (LibreOffice) -> PNG по страницам (PyMuPDF).

    python docs/render_preview.py docs/presentation.pptx <out_dir>
    python docs/render_preview.py docs/report.pdf <out_dir>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF

SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"


def main():
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pdf = src
    if src.suffix.lower() != ".pdf":
        subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(out), str(src)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        pdf = out / (src.stem + ".pdf")
    doc = fitz.open(pdf)
    for old in out.glob(f"{src.stem}-*.png"):
        old.unlink()
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=110).save(out / f"{src.stem}-{i:02d}.png")
    print(f"{len(doc)} страниц -> {out}")


if __name__ == "__main__":
    main()
