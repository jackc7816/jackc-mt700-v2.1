"""Evidence-first document reading for the V2 workflow."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path


def _with_evidence_ids(evidence: list[dict]) -> list[dict]:
    """Give every OCR block a stable identifier for evidence-bound extraction."""
    output = []
    for index, item in enumerate(evidence, start=1):
        record = dict(item)
        record.setdefault("id", f"p{record.get('page', 1)}-e{index}")
        output.append(record)
    return output


def _text_lines(text: str, page: int, source: str) -> list[dict]:
    return [
        {"page": page, "source": source, "location": f"第 {page} 页第 {index} 行", "text": line}
        for index, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]


def read_document_evidence(filename: str, content: bytes) -> dict:
    """Read text layer first, then local OCR when it is available.

    This function never turns an OCR failure into a documentary discrepancy.
    """
    suffix = Path(filename).suffix.lower()
    tencent_error = ""
    if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
        try:
            from tencent_ocr import recognize_image

            text, evidence = recognize_image(content, filename)
            if text:
                return {"status": "已使用腾讯云高精度 OCR；请人工确认", "text": text, "evidence": _with_evidence_ids(evidence)}
        except Exception as exc:
            # Continue to the offline fallback, but never conceal why the
            # preferred cloud reader was unavailable.  The exception message
            # contains no request headers or credentials.
            detail = " ".join(str(exc).split()) or type(exc).__name__
            tencent_error = f"腾讯云 OCR 未返回结果：{detail[:180]}"
    if suffix in {".txt", ".csv"}:
        text = content.decode("utf-8", errors="replace")
        return {"status": "文本已读取", "text": text, "evidence": _with_evidence_ids(_text_lines(text, 1, "text"))}

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            pages = [(index, page.extract_text() or "") for index, page in enumerate(reader.pages, start=1)]
            text = "\n".join(value for _, value in pages).strip()
            if text:
                evidence = [item for page, value in pages for item in _text_lines(value, page, "pdf_text")]
                return {"status": "已读取 PDF 文字层", "text": text, "evidence": _with_evidence_ids(evidence)}
        except Exception:  # Invalid/encrypted PDFs are handled as an unreadable input.
            pass

    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        return {"status": "该文件格式未提供可读文本", "text": "", "evidence": []}

    try:
        from local_ocr import recognize

        lines = recognize(content, suffix)
        text = "\n".join(str(line.get("text", "")) for line in lines if line.get("text")).strip()
        evidence = [
            {
                "page": line.get("page", "1"),
                "source": "local_ocr",
                "location": "页 {page}，x={x:.0f}，y={y:.0f}".format(
                    page=line.get("page", "1"), x=float(line.get("x", 0)), y=float(line.get("y", 0))
                ),
                "text": str(line.get("text", "")),
                # Keep provider-neutral geometry: downstream extraction must use
                # the same relative anchors for Tencent and local OCR evidence.
                "x": line.get("x", 0), "y": line.get("y", 0),
                "w": line.get("w", 0), "h": line.get("h", 0),
                "confidence": line.get("confidence", ""),
            }
            for line in lines
            if line.get("text")
        ]
        suffix_note = f"（{tencent_error}，已回退本地 OCR）" if tencent_error else ""
        return {"status": "已使用本地 OCR；请人工确认" + suffix_note, "text": text, "evidence": _with_evidence_ids(evidence)}
    except Exception as exc:
        prefix = f"{tencent_error}；" if tencent_error else ""
        return {
            "status": f"{prefix}无法读取文字：{type(exc).__name__}。请人工填写，系统不会据此判定不符。",
            "text": "",
            "evidence": [],
        }
