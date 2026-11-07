"""Document classification/extraction adapter for the teaching submission centre.

Set DOCUMENT_CLASSIFICATION_API_URL and DOCUMENT_EXTRACTION_API_URL to connect
an approved document-AI service. Local extraction uses PDF text or Windows OCR.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime
from io import BytesIO
import json
import os
import re
import unicodedata
from pathlib import Path
from subprocess import TimeoutExpired
from urllib.error import URLError
from urllib.request import Request, urlopen
from local_ocr import recognize
from structured_extraction import extract_structured, flat_structured_fields


MODEL_DOCUMENT_TYPES = {
    "商业发票": "commercial_invoice", "装箱单": "packing_list", "提单／运输单据": "bill_of_lading",
    "保险单／保险凭证": "insurance_certificate", "原产地证": "certificate_of_origin",
    "受益人证明": "beneficiary_certificate", "其他／需人工识别": "other",
}
EXTRACTION_PIPELINE_VERSION = "2026-11-06-multipage-table-v1"


def _deepseek_model_type(document_type: str, text: str) -> str | None:
    """Keep the finer document subtype when the V2 UI uses one shared label."""
    if document_type == "检验证书":
        # Suppliers often title this document "CERTIFICATE OF WEIGHT AND
        # ANALYSIS".  It contains the Cast No./chemical-analysis table and
        # must therefore use the analysis schema, not the generic inspection
        # schema.
        return "certificate_of_analysis" if re.search(r"CERTIFICATE\s+OF\s+(?:WEIGHT\s+AND\s+)?ANALYSIS|ANALYSIS\s+CERTIFICATE", text or "", re.I) else "inspection_certificate"
    if document_type == "提单／运输单据" and re.search(r"AIR\s*WAYBILL|AIR\s*WAY\s*BILL", text or "", re.I):
        return "air_waybill"
    return MODEL_DOCUMENT_TYPES.get(document_type)


def _activate_deepseek_user_key() -> str:
    """Use a valid Windows user key when Streamlit inherited a stale key.

    The key is only placed in this process environment; it is never logged,
    returned, or written to a file.
    """
    current = os.getenv("DEEPSEEK_API_KEY", "").strip()
    # An empty value is a deliberate "local-only" configuration and must not
    # silently opt into a cloud call.  Only repair a non-empty stale value.
    if not current or current.startswith("sk-") or os.name != "nt":
        return current
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            saved, _ = winreg.QueryValueEx(key, "DEEPSEEK_API_KEY")
        saved = str(saved).strip()
        if saved.startswith("sk-"):
            os.environ["DEEPSEEK_API_KEY"] = saved
            return saved
    except OSError:
        pass
    return current


DOCUMENT_TYPES = [
    "商业发票", "提单／运输单据", "装箱单", "保险单／保险凭证", "原产地证",
    "检验证书", "受益人证明", "其他／需人工识别",
]


def _text_from_bytes(content: bytes, filename: str) -> str:
    """Read local text files and text-layer PDFs; never perform OCR locally."""
    if filename.lower().endswith((".txt", ".csv", ".json", ".xml")):
        return content.decode("utf-8", errors="replace")
    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            from pypdf.errors import EmptyFileError, PdfReadError, PdfStreamError
        except ImportError:
            return ""

        try:
            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except (OSError, ValueError, EmptyFileError, PdfReadError, PdfStreamError):
            return ""
    return ""


def _post_json(endpoint: str, payload: dict) -> dict:
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:  # nosec: URL comes from local deployment configuration
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _heuristic_type(filename: str, text: str) -> str:
    patterns = {
        "保险单／保险凭证": ("INSURANCE POLICY", "CERTIFICATE OF INSURANCE", "INSURANCE", "保险单", "保险凭证"),
        "商业发票": ("COMMERCIAL INVOICE", "INVOICE", "FAKTUR", "商业发票", "发票"),
        "提单／运输单据": ("BILL OF LADING", "OCEAN BILL", "AIR WAYBILL", "KONOSEMEN", "B/L", "海运提单", "空运提单", "提单"),
        "装箱单": ("PACKING LIST", "PACKING", "装箱单"),
        "原产地证": ("CERTIFICATE OF ORIGIN", "原产地证", "产地证"),
        "检验证书": ("CERTIFICATE OF ANALYSIS", "INSPECTION CERTIFICATE", "SURVEY REPORT", "分析证书", "检验证书", "质检证书"),
    }
    for source in (filename.upper(), text[:4000].upper()):
        for doc_type, tokens in patterns.items():
            if any(token in source for token in tokens if token not in {"INVOICE", "PACKING", "B/L"}):
                return doc_type
        for token, doc_type in (("INVOICE", "商业发票"), ("PACKING", "装箱单"), ("B/L", "提单／运输单据")):
            if token in source:
                return doc_type
        # Subsequent invoice pages often omit the title and invoice number.
        # The finance footer below is characteristic of this invoice layout;
        # require several anchors to avoid labelling an arbitrary payment page
        # as a commercial invoice.
        if all(token in source for token in ("TOTAL DUE", "LESS PREPAYMENT", "LME", "PREMIUM")):
            return "商业发票"

        # A multi-page container packing list commonly prints its heading only
        # on the first page.  Later pages still carry the compact container
        # subtotal footer.  Treat that distinctive combination as a packing
        # list so a batch upload does not silently omit a continuation page.
        # Requiring the total plus two further footer labels avoids classifying
        # a transport document merely because it mentions a container.
        packing_continuation = ("CONTAINER TOTAL" in source and sum(
            token in source for token in ("SEAL", "TARE KG", "NET WT", "GROSS WT", "NO OF PIECES")
        ) >= 2)
        if packing_continuation:
            return "装箱单"
    return "其他／需人工识别"


def classify_document(filename: str, content: bytes, extracted_text: str = "", evidence: list[dict] | None = None) -> dict:
    """Classify from OCR evidence, optionally using DeepSeek, then local rules.

    ``extracted_text`` lets a caller reuse local OCR output.  This is important
    for camera/WeChat filenames, which normally contain no useful document-type
    clue.  It is optional to keep V1 callers compatible.
    """
    text = extracted_text or _text_from_bytes(content, filename)
    if _activate_deepseek_user_key():
        try:
            from deepseek_classifier import classify_document as deepseek_classify

            model_result = deepseek_classify(filename, text, evidence)
            type_map = {
                "packing_list": "装箱单", "commercial_invoice": "商业发票", "bill_of_lading": "提单／运输单据", "air_waybill": "提单／运输单据",
                "insurance_certificate": "保险单／保险凭证", "certificate_of_origin": "原产地证", "certificate_of_analysis": "检验证书",
                "inspection_certificate": "检验证书", "beneficiary_certificate": "受益人证明", "beneficiary_statement": "受益人证明",
            }
            model_type = model_result["document_type"]
            alternatives = [type_map.get(item, "其他／需人工识别") for item in model_result["alternative_types"]]
            return {
                "document_type": type_map.get(model_type, "其他／需人工识别"),
                "source": "DeepSeek 分类建议（待人工确认）",
                "confidence": f"模型置信度 {model_result['confidence']:.0%}；待人工确认",
                "model_document_type": model_type,
                "alternative_types": list(dict.fromkeys(alternatives)),
                "reason": model_result["reason"],
                "needs_human_confirmation": True,
            }
        except Exception as exc:
            detail = " ".join(str(exc).split())[:180]
            suffix = f"（{detail}）" if detail else ""
            deepseek_error = f"DeepSeek 分类不可用，已本地推断：{type(exc).__name__}{suffix}"
    else:
        deepseek_error = ""

    endpoint = os.getenv("DOCUMENT_CLASSIFICATION_API_URL", "").strip()
    if endpoint:
        try:
            result = _post_json(endpoint, {"filename": filename, "content_base64": base64.b64encode(content).decode("ascii")})
            doc_type = result.get("document_type") or result.get("classification")
            if doc_type in DOCUMENT_TYPES:
                return {"document_type": doc_type, "source": "分类 API", "confidence": result.get("confidence", "未提供")}
            return {"document_type": "其他／需人工识别", "source": "分类 API 返回未知类型", "confidence": result.get("confidence", "未提供")}
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return {"document_type": _heuristic_type(filename, text), "source": f"分类 API 不可用，已本地推断：{type(exc).__name__}", "confidence": "待人工确认"}
    source = "本地文件名/OCR/文本推断（未配置分类 API）" if extracted_text else "本地文件名/文本推断（未配置分类 API）"
    if deepseek_error:
        source = f"{source}；{deepseek_error}"
    return {"document_type": _heuristic_type(filename, text), "source": source, "confidence": "待人工确认"}


def _first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.I | re.M)
    return " ".join(match.group(1).split()) if match else ""


def _next_line_after_label(text: str, labels: tuple[str, ...]) -> str:
    """Read layouts where a field label occupies one line and its value the next."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    normalized_labels = {label.upper() for label in labels}
    for index, line in enumerate(lines):
        if line.upper().strip(" :") not in normalized_labels:
            continue
        for candidate in lines[index + 1:]:
            if candidate:
                return candidate
    return ""


def _invoice_total(text: str) -> tuple[str, str]:
    """Prefer a labelled invoice total over the first unit-price amount in a PDF."""
    match = re.search(
        r"(?is)\b(?:INVOICE\s+TOTAL|TOTAL\s+INVOICE\s+AMOUNT)\b[^\n]*\n\s*([A-Z]{3})\s*([0-9][0-9,\.]*)",
        text,
    )
    if match:
        return match.group(1), match.group(2)
    match = re.search(r"\b([A-Z]{3})\s*([0-9][0-9,\.]*)", text.upper())
    return match.groups() if match else ("", "")


SCHEMAS = {
    "商业发票": ("发票号", "发票币种", "发票金额", "发票出口商", "发票进口商", "发票货物描述", "发票数量", "发票日期"),
    "提单／运输单据": ("提单号", "装运日", "提单装货港", "提单卸货港", "提单货物描述", "提单数量"),
    "装箱单": ("装箱单号", "装箱单货物描述", "装箱单数量", "装箱单装货港", "装箱单卸货港", "装箱单毛重", "装箱单净重"),
    "保险单／保险凭证": ("保险单号", "保险币种", "保险金额", "保险货物描述", "保险数量", "保险索赔代理"),
    "原产地证": ("原产地证号", "原产地证货物描述", "原产地证数量", "原产国"),
    "检验证书": ("检验证书号", "检验证书货物描述"),
}
MONEY = r"\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2}"
PORT_LOADING = r"(?:PORT|PO\s*0F)\s*(?:OF\s*)?LOADING|装货港"
PORT_DISCHARGE = r"PO(?:RT|戊)\s+OF\s+DISCHARGE|卸货港"


def _clean(value: str) -> str:
    if re.match(r"^(?:TBC|N/A|TO\s*(?:BE|BÉ)?\s*(?:PROVIDED|CONFIRMED|QUOTED)|NOT PROVIDED)(?:\b|$)", value, re.I):
        return ""
    return value


def _normalize_ocr(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"(?<=\d)[ \t]*([,.])[ \t]*(?=\d)", r"\1", text)


def _label_value(text: str, label: str) -> str:
    return _clean(_first(rf"^[ \t]*(?:{label})[ \t]*[:.#]?[ \t]*(?:\n[ \t]*)?([^\n]+)", text))


def _beside(lines: list[dict], label: str, value_pattern: str) -> str:
    """Find a value on the same row as its label, even in a separate OCR column."""
    for anchor in lines:
        text = _normalize_ocr(anchor["text"]).strip()
        same_line = re.fullmatch(rf"(?:{label})[ \t]*[:. ]*[ \t]+({value_pattern})", text, re.I)
        if same_line:
            return same_line[1]
        if not re.fullmatch(rf"(?:{label})[ :.]*", text, re.I):
            continue
        row = [
            line for line in lines
            if line["page"] == anchor["page"]
            and line["x"] >= anchor["x"] + anchor["w"] - 4
            and abs(line["y"] - anchor["y"]) <= max(anchor["h"], line["h"]) * 0.65
            and re.fullmatch(value_pattern, _normalize_ocr(line["text"]).strip(), re.I)
        ]
        if row:
            return _normalize_ocr(min(row, key=lambda line: line["x"])["text"]).strip()
    return ""


def _after_table_label(text: str, label: str) -> str:
    """Read common table layouts where a numbered label and value span lines."""
    match = re.search(
        rf"(?im)(?:^|\n)\s*\d{{0,2}}\.?\s*(?:{label})\s*(?:\n|\s{{2,}})([A-Z][A-Z0-9 ,./&'()-]{{2,}})",
        text,
    )
    return _clean(" ".join(match.group(1).split())) if match else ""


def _below_label(lines: list[dict], label: str, value_pattern: str) -> str:
    """Use OCR coordinates when a table label is above its value, not beside it."""
    for anchor in lines:
        anchor_text = _normalize_ocr(str(anchor.get("text", ""))).strip()
        if not re.search(label, anchor_text, re.I):
            continue
        candidates = []
        for line in lines:
            value = _normalize_ocr(str(line.get("text", ""))).strip()
            if line.get("page") != anchor.get("page") or not re.fullmatch(value_pattern, value, re.I):
                continue
            if float(line.get("x", 0)) < float(anchor.get("x", 0)) - 60:
                continue
            vertical_gap = float(line.get("y", 0)) - float(anchor.get("y", 0))
            if 0 < vertical_gap <= max(float(anchor.get("h", 0)) * 3.5, 110):
                candidates.append((vertical_gap, float(line.get("x", 0)), value))
        if candidates:
            return min(candidates)[2]
    return ""


def _right_of_label(lines: list[dict], label: str) -> str:
    """Join small adjacent OCR blocks, such as `11` + `MAY` + `2024`."""
    for anchor in lines:
        if not re.search(label, _normalize_ocr(str(anchor.get("text", ""))), re.I):
            continue
        anchor_right = float(anchor.get("x", 0)) + float(anchor.get("w", 0)) - 4
        nearby = [
            line for line in lines
            if line.get("page") == anchor.get("page")
            and float(line.get("x", 0)) >= anchor_right
            and abs(float(line.get("y", 0)) - float(anchor.get("y", 0))) <= max(float(anchor.get("h", 0)) * 1.6, 45)
            and str(line.get("text", "")).strip()
        ]
        if nearby:
            return " ".join(str(line["text"]).strip() for line in sorted(nearby, key=lambda item: float(item.get("x", 0))))
    return ""


def _first_company_before(lines: list[dict], anchor_label: str) -> str:
    """Find the issuer line above a document title in common invoice layouts."""
    anchor_y = min(
        (float(line.get("y", 0)) for line in lines if re.search(anchor_label, str(line.get("text", "")), re.I)),
        default=float("inf"),
    )
    candidates = [
        line for line in lines
        if float(line.get("y", 0)) < anchor_y
        and re.search(r"\b(?:PTE\.?\s*LTD\.?|LTD\.?|LIMITED|CO\.?|INC\.?)\b", str(line.get("text", "")), re.I)
    ]
    return str(min(candidates, key=lambda line: (float(line.get("y", 0)), float(line.get("x", 0))))["text"]).strip() if candidates else ""


def _last_invoice_money(text: str) -> tuple[str, str]:
    """Invoice tables commonly show a unit price before the final amount."""
    matches = re.findall(rf"\b([A-Z]{{3}})\s*({MONEY})\b", text.upper())
    return matches[-1] if matches else ("", "")


def _shipment_date(text: str) -> str:
    match = re.search(r"(?i)(?:DATE\s+LADEN\s+ON\s+BOARD|ON[ -]?BOARD\s+DATE|SHIPMENT\s+DATE)[^\dA-Z]*(\d{1,2}\s+[A-Z]{3,9}\s+\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", text)
    raw = (match.group(1) if match else text).strip().upper()
    for pattern in ("%d %B %Y", "%d %b %Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(raw.title() if "%B" in pattern or "%b" in pattern else raw, pattern).date().isoformat()
        except ValueError:
            continue
    return raw


def _bill_goods(text: str) -> str:
    """Prefer commodity lines over the printed 'Description of Goods' heading."""
    match = re.search(r"(?im)^\s*(PRIMARY\s+ALUMINIUM(?:\s+INGOT)?[^\n;]*)", text)
    if match:
        return " ".join(match.group(1).split())
    lines = [" ".join(line.split()) for line in text.splitlines()]
    for index, line in enumerate(lines):
        if "DESCRIPTION OF GOODS" not in line.upper():
            continue
        candidates = []
        for candidate in lines[index + 1:index + 8]:
            upper = candidate.upper()
            if not candidate or any(token in upper for token in ("GROSS WEIGHT", "MEASUREMENT", "FREIGHT", "DESCRIPTION OF CONTENTS")):
                continue
            candidates.append(candidate)
            if len(candidates) == 2:
                break
        if candidates:
            return " ".join(candidates)
    return ""


def _scan_fields(doc_type: str, lines: list[dict], text: str) -> dict:
    quantity = _first(r"^\s*QUANTITY\s*[:.]?\s*([\d,.]+)\b", text)
    package_count = _first(r"\b(\d{1,6})\s+(?:BUNDLES?|PIECES?|PACKAGES?|CARTONS?|CTNS?)\b", text)
    goods = _bill_goods(text) or _label_value(text, r"DESCRIPTION\s+OF\s+GOODS|GOODS DESCRIPTION|货物描述")
    if not goods:
        goods = _first(r"^[ \t]*(PRIMARY ALUMINIUM(?:[ \t]*\n[ \t]*| )[\w| ]*N\s*GOT[^\n;]*)", text)
    # OCR guesses for model codes must remain visible for user verification.
    goods = re.split(r"\s{2,}|;", goods)[0]
    if doc_type == "商业发票":
        invoice_no = _below_label(lines, r"INVOICE\s+(?:NO\.?|NUMBER)", r"[A-Z0-9-]{4,}") \
            or _label_value(text, r"(?:INVOICE|LNVOICE)\s+(?:NO|NUMBER|REFERENCE)\.?|发票号")
        invoice_date = _shipment_date(
            _below_label(lines, r"INVOICE\s+DATE", r"\d{1,2}\s+[A-Z]{3,9}\s+\d{4}")
            or _right_of_label(lines, r"INVOICE\s+DATE")
            or _label_value(text, r"(?:INVOICE|DOCUMENT)\s+DATE")
        )
        currency, amount = _last_invoice_money(text)
        if not amount:
            amount = _beside(lines, r"(?:INVOICE|LNVOICE)\s+TOTAL|TOTAL INVOICE AMOUNT|AMOUNT", MONEY)
        if not currency and amount:
            currency_match = re.search(rf"\b([A-Z]{{3}})\s*{re.escape(amount)}\b", text.upper())
            currency = currency_match.group(1) if currency_match else ""
        exporter = _first_company_before(lines, r"COMMERCIAL\s+INVOICE") or _label_value(text, r"SELLER|EXPORTER NAME|COMPANY NAME/ADDRESS")
        importer = _below_label(lines, r"SOLD\s+TO|BUYER|IMPORTER", r".+") \
            or _clean(_first(r"^SAME AS CONSIGNEE[ \t]*:[ \t]*([^\n]+)", text)) \
            or _label_value(text, r"BUYER|IMPORTER NAME")
        return {
            "发票号": invoice_no,
            "发票币种": currency or _beside(lines, r"CURRENCY CODE", r"[A-Z]{3}"),
            "发票金额": amount,
            "发票出口商": exporter,
            "发票进口商": importer,
            "发票货物描述": goods, "发票数量": quantity or package_count,
            "发票日期": invoice_date,
        }
    if doc_type == "提单／运输单据":
        bill_number = _beside(lines, r"(?:B/?L|&L|BL|BILL\s+OF\s+LADING)\s*(?:NO\.?|NUMBER)?", r"[A-Z0-9][A-Z0-9-]{4,}") \
            or _below_label(lines, r"BILL\s+OF\s+LADING\s*(?:NO\.?|NUMBER)?", r"[A-Z0-9][A-Z0-9-]{4,}") \
            or _label_value(text, r"(?:B/?L|&L|BL|BILL\s+OF\s+LADING)\s*(?:NO\.?|NUMBER)?|提单号")
        bill_number = re.sub(r"\s+", "", bill_number)
        loading = _below_label(lines, r"PORT\s+OF\s+LOADING", r"[A-Z][A-Z0-9 ,./&'()-]{2,}") or _after_table_label(text, r"PORT\s+OF\s+LOADING") or _label_value(text, PORT_LOADING)
        discharge = _below_label(lines, r"PORT\s+OF\s+DISCHARGE", r"[A-Z][A-Z0-9 ,./&'()-]{2,}") or _after_table_label(text, r"PORT\s+OF\s+DISCHARGE") or _label_value(text, PORT_DISCHARGE)
        return {
            "提单号": bill_number,
            "装运日": _shipment_date(_right_of_label(lines, r"DATE\s+LADEN\s+ON\s+BOARD") or text),
            "提单装货港": loading, "提单卸货港": discharge,
            "提单货物描述": goods, "提单数量": quantity or package_count,
        }
    if doc_type == "装箱单":
        return {
            "装箱单号": _label_value(text, r"PACKING LIST\s+(?:NO\.?|NUMBER)"),
            "装箱单货物描述": goods, "装箱单数量": quantity,
            "装箱单装货港": _label_value(text, PORT_LOADING), "装箱单卸货港": _label_value(text, PORT_DISCHARGE),
            "装箱单毛重": _label_value(text, r"TOTAL GROSS WEIGHT"),
            "装箱单净重": _label_value(text, r"TOTAL NET WEIGHT"),
        }
    if doc_type == "保险单／保险凭证":
        insured = re.search(rf"\b([A-Z]{{3}})\s+({MONEY})\s*\(\s*110\s*%", text)
        if not insured:
            insured = re.search(rf"(?:INSURED VALUE|SUM INSURED)[ :.\n]+([A-Z]{{3}})\s+({MONEY})", text, re.I)
        return {
            "保险单号": _label_value(text, r"(?:POLICY|CERTIFICATE)\s+(?:NO\.?|NUMBER)"),
            "保险币种": insured[1] if insured else "", "保险金额": insured[2] if insured else "",
            "保险货物描述": goods,
            "保险数量": quantity or _first(r";[ \t]*([\d,.]+)[ \t]+M[ \t]*TS\b", text),
        }
    if doc_type == "原产地证":
        return {
            "原产地证号": _label_value(text, r"CERTIFICATE\s+(?:NO\.?|NUMBER)"),
            "原产地证货物描述": goods, "原产地证数量": quantity,
            "原产国": _label_value(text, r"COUNTRY\s+OF\s+ORIGIN|原产国"),
        }
    if doc_type == "检验证书":
        return {"检验证书号": _label_value(text, r"REPORT\s+(?:NO\.?|NUMBER)"), "检验证书货物描述": goods}
    return {}


def _local_extract(doc_type: str, filename: str, content: bytes) -> dict:
    text = _text_from_bytes(content, filename)
    fields = {"文件名": filename, "文档类型": doc_type}
    fields.update(dict.fromkeys(SCHEMAS.get(doc_type, ()), ""))
    if not text:
        suffix = Path(filename).suffix.lower()
        prefix = "PDF 未读取到文字层。" if suffix == ".pdf" else ""
        if suffix in {".pdf", ".png", ".jpg", ".jpeg"}:
            try:
                lines = recognize(content, suffix)
                if lines:
                    text = _normalize_ocr("\n".join(line["text"] for line in lines))
                    fields.update(_scan_fields(doc_type, lines, text))
                    fields["可读文本预览"] = text
                    fields["提取状态"] = "已使用本地 OCR。请对照原件核对识别值，补填空字段。"
                    return fields
                detail = "本地 OCR 未识别出文字。"
            except ImportError:
                detail = "本地 OCR 缺少依赖，请安装 requirements.txt 中的依赖。"
            except TimeoutExpired:
                detail = "本地 OCR 超时，请拆分文件后重试。"
            except (OSError, ValueError, RuntimeError):
                detail = "本地 OCR 无法处理此文件。"
        else:
            detail = "没有可读取的文本。"
        fields["提取状态"] = prefix + detail + " 可直接填写下方字段。"
        return fields
    if doc_type == "商业发票":
        currency, amount = _invoice_total(text)
        fields.update({
            "发票号": _next_line_after_label(text, ("INVOICE REFERENCE", "INVOICE NO", "INVOICE NUMBER", "发票号")) or _first(r"(?m)^(?:INVOICE(?:[ \t]*(?:NO\.?|NUMBER))?|发票号)[ \t]*[:#]?[ \t]*([^\n]+)", text),
            "发票币种": currency,
            "发票金额": amount,
            "发票出口商": _next_line_after_label(text, ("EXPORTER / SELLER", "SELLER", "EXPORTER", "出口商")) or _first(r"(?:SELLER|EXPORTER|出口商)\s*[:#]?\s*([^\n]+)", text),
            "发票进口商": _next_line_after_label(text, ("BUYER / IMPORTER", "BUYER", "IMPORTER", "进口商")) or _first(r"(?:BUYER|IMPORTER|进口商)\s*[:#]?\s*([^\n]+)", text),
            "发票货物描述": _first(r"(?:DESCRIPTION|GOODS|货物)\s*[:#]?\s*([^\n]+)", text),
            "发票数量": _first(r"(?:QUANTITY|数量)\s*[:#]?\s*([^\n]+)", text),
        })
    elif doc_type == "提单／运输单据":
        fields.update({"提单号": _first(r"(?:B/?L\s*(?:NO\.?|NUMBER)?|BILL\s+OF\s+LADING\s*(?:NO\.?|NUMBER)?|提单号)\s*[:#]?\s*([^\n]+)", text), "装运日": _first(r"(?:SHIPPED\s+ON\s+BOARD|SHIPMENT\s+DATE|装运日)\s*[:#]?\s*([^\n]+)", text), "提单装货港": _first(r"(?:PORT\s+OF\s+LOADING|装货港)\s*[:#]?\s*([^\n]+)", text), "提单卸货港": _first(r"(?:PORT\s+OF\s+DISCHARGE|卸货港)\s*[:#]?\s*([^\n]+)", text), "提单货物描述": _first(r"(?:DESCRIPTION\s+OF\s+GOODS|货物)\s*[:#]?\s*([^\n]+)", text), "提单数量": _first(r"(?:QUANTITY|数量)\s*[:#]?\s*([^\n]+)", text)})
    else:
        fields.update(_scan_fields(doc_type, [], text))
    for key in SCHEMAS.get(doc_type, ()):
        fields[key] = _clean(fields[key])
    fields["可读文本预览"] = text
    source_label = "本地 PDF 文字层规则提取" if filename.lower().endswith(".pdf") else "本地文本规则提取"
    fields["提取状态"] = f"{source_label}（未配置抽取 API），请人工确认。"
    return fields


def extract_from_read_text(filename: str, document_type: str, text: str, evidence: list[dict] | None = None, source_label: str = "OCR") -> dict:
    """Extract fields from already-read text without invoking a second OCR pass."""
    fields = {"文件名": filename, "文档类型": document_type}
    fields.update(dict.fromkeys(SCHEMAS.get(document_type, ()), ""))
    fields.update(_scan_fields(document_type, evidence or [], text))
    for key in SCHEMAS.get(document_type, ()):
        fields[key] = _clean(fields.get(key, ""))
    fields["可读文本预览"] = text
    fields["提取状态"] = f"已使用{source_label}提取；请对照原件核对字段。"
    return fields


def extract_document(filename: str, content: bytes, document_type: str, ocr_text: str | None = None, ocr_evidence: list[dict] | None = None, source_label: str = "OCR") -> dict:
    if ocr_text:
        return extract_from_read_text(filename, document_type, ocr_text, ocr_evidence, source_label)
    endpoint = os.getenv("DOCUMENT_EXTRACTION_API_URL", "").strip()
    if endpoint:
        try:
            result = _post_json(endpoint, {"filename": filename, "document_type": document_type, "content_base64": base64.b64encode(content).decode("ascii")})
            fields = result.get("fields", result)
            if isinstance(fields, dict):
                fields = {str(k): str(v) for k, v in fields.items()}
                fields["提取状态"] = "抽取 API 返回；请人工确认。"
                return fields
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
            pass
    return _local_extract(document_type, filename, content)


def _fields_from_structured(filename: str, document_type: str, ocr_text: str, structured: dict) -> dict:
    """Keep a single flat UI view without discarding the evidence-rich result."""
    fields = {"文件名": filename, "文档类型": document_type, "可读文本预览": ocr_text}
    structured["extraction_pipeline_version"] = EXTRACTION_PIPELINE_VERSION
    fields.update({key: detail.get("value", "") for key, detail in structured.get("fields", {}).items() if isinstance(detail, dict)})
    model = structured.get("model", "deepseek-flash")
    page_count = len(structured.get("source_pages", []))
    page_note = f"，已合并 {page_count} 页" if page_count > 1 else ""
    fields["提取状态"] = f"已使用 DeepSeek V4.1 Flash（{model}）结构化提取{page_note}；所有字段均需人工确认。"
    for key, value in flat_structured_fields(structured).items():
        if value:
            fields[key] = value
    return fields


def _apply_evidence_safe_repairs(structured: dict, model_type: str, ocr_text: str, evidence: list[dict] | None) -> dict:
    """Apply only structural, evidence-backed corrections after model extraction.

    This does not invent or alter documentary values.  It fills an omitted
    analysis order number only when that exact value is in an OCR evidence
    block, removes a known table-column separator from a B/L seal, and
    suppresses a known OCR misspelling rather than presenting it as fact.
    """
    fields = structured.get("fields", {}) if isinstance(structured, dict) else {}
    rows = structured.get("table_data", []) if isinstance(structured, dict) else []
    safe_evidence = evidence or []
    if model_type == "certificate_of_analysis" and isinstance(fields.get("order_number"), dict):
        # The analysis form has an empty "TOMAGO ORDER NO" next to a
        # populated "PART'T ORDER NO".  Only accept a numeric value so that
        # nearby headings such as MANIFEST cannot be mistaken for an order.
        match = re.search(r"(?:PART'?T|PART|TOMAGO)\s+ORDER\s+NO\s*[:=]?\s*(?:\(\d+\)\s*)?(\d{4,})", ocr_text or "", re.I)
        if match:
            value = match.group(1)
            item = next((row for row in safe_evidence if value.upper() in str(row.get("text", "")).upper()), None)
            if item:
                fields["order_number"] = {
                    "value": value, "confidence": 0.95, "source_page": item.get("page"),
                    "source_text": str(item.get("text", "")), "evidence_ids": [str(item.get("id"))],
                    "needs_human_confirmation": True,
                }
    if model_type == "bill_of_lading":
        for row in rows if isinstance(rows, list) else []:
            seal = row.get("seal_number") if isinstance(row, dict) else ""
            if isinstance(seal, str) and re.fullmatch(r"/[A-Z]\d{6}", seal.upper()):
                row["seal_number"] = seal[1:]
    for detail in fields.values() if isinstance(fields, dict) else []:
        if isinstance(detail, dict) and "ALUMINTUM" in str(detail.get("value", "")).upper():
            detail.update({"value": "", "confidence": 0, "source_page": None, "source_text": "", "evidence_ids": [], "needs_human_confirmation": True})
            structured.setdefault("warnings", []).append("OCR 将 ALUMINIUM 识别为 ALUMINTUM；为避免错误字段，已留空待人工确认。")
    return structured


def merge_structured_extractions(document_type: str, filename: str, ocr_text: str, page_results: list[dict]) -> tuple[dict, dict]:
    """Merge page-level model outputs without losing later-page table rows."""
    usable = [item for item in page_results if isinstance(item, dict)]
    if not usable:
        return {}, {"document_type": document_type, "fields": {}, "table_data": [], "extraction_status": "pending_confirmation"}
    structured = deepcopy(usable[0])
    fields = structured.setdefault("fields", {})
    table_data: list[dict] = []
    table_index: dict[tuple, dict] = {}
    warnings: list[str] = []
    unresolved: list[str] = []
    source_pages: set[int] = set()
    for result in usable:
        for key, detail in result.get("fields", {}).items():
            if key not in fields or not isinstance(fields.get(key), dict) or not fields[key].get("value"):
                if isinstance(detail, dict) and detail.get("value"):
                    fields[key] = deepcopy(detail)
        for row in result.get("table_data", []):
            if not isinstance(row, dict):
                continue
            primary = str(row.get("container_number") or row.get("cast_number") or "").strip().upper()
            identity = ("primary", primary) if primary else tuple(sorted((key, str(value)) for key, value in row.items() if key not in {"evidence_ids", "source_page"}))
            if identity and identity not in table_index:
                record = deepcopy(row)
                table_index[identity] = record
                table_data.append(record)
            elif identity in table_index:
                existing = table_index[identity]
                for key, value in row.items():
                    if key in {"evidence_ids", "source_page"}:
                        continue
                    if not existing.get(key) and value:
                        existing[key] = value
                existing_ids = list(dict.fromkeys([*existing.get("evidence_ids", []), *row.get("evidence_ids", [])]))
                existing["evidence_ids"] = existing_ids
        warnings.extend(str(item) for item in result.get("warnings", []) if item)
        unresolved.extend(str(item) for item in result.get("unresolved_items", []) if item)
        for page in [row.get("source_page") for row in result.get("table_data", []) if isinstance(row, dict)]:
            if isinstance(page, int):
                source_pages.add(page)
        for detail in result.get("fields", {}).values():
            if isinstance(detail, dict) and isinstance(detail.get("source_page"), int):
                source_pages.add(detail["source_page"])
    structured["table_data"] = table_data
    structured["warnings"] = list(dict.fromkeys(warnings))
    structured["unresolved_items"] = list(dict.fromkeys(unresolved))
    structured["source_pages"] = sorted(source_pages)
    structured["provider"] = "DeepSeek" if all(item.get("provider") == "DeepSeek" for item in usable) else structured.get("provider", "local")
    structured["extraction_pipeline_version"] = EXTRACTION_PIPELINE_VERSION
    return _fields_from_structured(filename, document_type, ocr_text, structured), structured


def extract_document_with_structure(filename: str, content: bytes, document_type: str, ocr_text: str | None = None, ocr_evidence: list[dict] | None = None, source_label: str = "OCR", preserve_page_context: bool = False) -> tuple[dict, dict]:
    """Use DeepSeek for evidence-bound extraction, with a safe local fallback."""
    model_type = _deepseek_model_type(document_type, ocr_text or "")
    if ocr_text and model_type and _activate_deepseek_user_key():
        try:
            from deepseek_structured_extractor import extract_document as deepseek_extract
            pages: dict[object, list[dict]] = {}
            for item in ocr_evidence or []:
                pages.setdefault(item.get("page", 1), []).append(item)
            # Multi-page re-extraction is page-bounded: later pages cannot be
            # silently truncated by one document-wide OCR/evidence limit.
            if len(pages) > 1 and not preserve_page_context:
                results = [_apply_evidence_safe_repairs(deepseek_extract(filename, model_type, "\n".join(str(item.get("text", "")) for item in evidence), evidence), model_type, "\n".join(str(item.get("text", "")) for item in evidence), evidence) for _, evidence in sorted(pages.items(), key=lambda pair: str(pair[0]))]
                return merge_structured_extractions(document_type, filename, ocr_text, results)
            structured = _apply_evidence_safe_repairs(deepseek_extract(filename, model_type, ocr_text, ocr_evidence), model_type, ocr_text, ocr_evidence)
            return _fields_from_structured(filename, document_type, ocr_text, structured), structured
        except Exception as exc:
            detail = " ".join(str(exc).split())[:180]
            model_warning = f"DeepSeek 结构化提取不可用，已回退本地规则：{detail or type(exc).__name__}"
    else:
        model_warning = ""
    fields = extract_document(filename, content, document_type, ocr_text, ocr_evidence, source_label)
    structured = extract_structured(document_type, ocr_text or fields.get("可读文本预览", ""), ocr_evidence or [])
    if model_warning:
        fields["提取状态"] = f"{fields.get('提取状态', '')}；{model_warning}"
        structured.setdefault("warnings", []).append(model_warning)
    for key, value in flat_structured_fields(structured).items():
        if value:
            fields[key] = value
    return fields, structured


def fields_to_record(fields: dict, document_type: str) -> dict:
    """Map user-confirmed extraction fields into the existing cross-review record."""
    ignored = {"文件名", "文档类型", "提取状态", "可读文本预览"}
    return {key: value for key, value in fields.items() if key not in ignored and str(value).strip() and document_type in DOCUMENT_TYPES}
