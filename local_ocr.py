"""Render scanned pages and run Windows OCR locally."""

from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def recognize(content: bytes, suffix: str) -> list[dict]:
    powershell = shutil.which("powershell.exe")
    if not powershell:
        raise RuntimeError("本机没有可用的 Windows OCR。")

    from PIL import Image, ImageOps

    with tempfile.TemporaryDirectory(prefix="trade_ocr_") as folder:
        directory = Path(folder)
        if suffix == ".pdf":
            import pypdfium2 as pdfium

            with pdfium.PdfDocument(content) as document:
                if len(document) > 20:
                    raise RuntimeError("本地 OCR 最多处理 20 页，请拆分文件。")
                for index in range(len(document)):
                    page = document[index]
                    try:
                        bitmap = page.render(scale=3500 / max(page.get_size()))
                        try:
                            bitmap.to_pil().save(directory / f"{index:03d}.png")
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
        else:
            with Image.open(BytesIO(content)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((3500, 3500))
                image.save(directory / "000.png")

        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(Path(__file__).with_name("windows_ocr.ps1")),
             "-ImageDirectory", str(directory)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise RuntimeError("Windows OCR 未能读取该文件，请核对文件和系统 OCR 语言包。")
        return json.loads(result.stdout)
