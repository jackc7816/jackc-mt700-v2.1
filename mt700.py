"""Small, deterministic parser for pasted or exported MT700/MT701 text.

It intentionally parses only common fields needed by the teaching review flow.
The original message remains available to the user for verification; a parsed
value is never treated as proof that the message is authentic or complete.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import re


FIELD_START = re.compile(r"(?m)^:([0-9]{2}[A-Z]?):")


def _clean(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.replace("\r", "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _date_from_yymmdd(value: str) -> str:
    try:
        return datetime.strptime(value[:6], "%y%m%d").date().isoformat()
    except ValueError:
        return ""


def _amount_32b(value: str) -> tuple[str, str]:
    match = re.search(r"([A-Z]{3})\s*([0-9][0-9,\.]*)", value.upper())
    if not match:
        return "", ""
    currency, raw_amount = match.groups()
    normalized = raw_amount.replace(".", "").replace(",", ".")
    try:
        return currency, str(Decimal(normalized))
    except InvalidOperation:
        return currency, ""


def _tolerance_39a(value: str) -> str:
    numbers = [int(number) for number in re.findall(r"\d{1,2}", value)]
    return str(max(numbers)) if numbers else ""


def _presentation_days(field_48: str, additional_conditions: str) -> str:
    """Extract a numeric post-shipment presentation period when explicitly stated.

    Field 48 takes priority. Some domestic-credit messages instead put the
    Field-48 instruction in 47A, so that is used only as a fallback.
    """
    for source in (field_48, additional_conditions):
        match = re.search(r"(?:^|\D)(\d{1,3})\s*(?:/\s*)?(?:DAYS?|HARI)\b", source.upper())
        if not match:
            match = re.search(r"(?:^|\D)(\d{1,3})\s*/\s*AFTER\s+SHIPMENT", source.upper())
        if match:
            return match.group(1)
    return ""


def _first_match(value: str, pattern: str) -> str:
    match = re.search(pattern, value, re.I | re.S)
    return " ".join(match.group(1).split()) if match else ""


def _key_goods_terms(value: str) -> dict[str, str]:
    """Extract display-only commercial terms from 45A; original 45A remains authoritative."""
    return {
        "合同号／依据": _first_match(value, r"(?:CONTRACT\s*(?:NO\.?|NUMBER)?|KONTRAK\s*NO\.?)\s*[:#]?\s*([^\n.]+)"),
        "贸易术语": _first_match(value, r"(?:TRADE\s+TERM|INCOTERM(?:S)?(?:\s*2020)?)\s*[:\-]?\s*([^\n.]+)"),
        "原产地": _first_match(value, r"(?:COUNTRY\s+OF\s+ORIGIN)\s*[:\-]?\s*([^\n.]+)"),
        "货物数量": _first_match(value, r"(?:TOTAL\s+QUANTITY|KUANTITAS\s+BARANG)\s*[:\-]?\s*([^\n.]+)"),
        "单价": _first_match(value, r"(?:UNIT\s+PRICE|HARGA)\s*[:\-]?\s*([^\n.]+)"),
    }


def _operational_conditions(value: str) -> dict[str, str]:
    upper = value.upper()
    return {
        "第三方单据": "允许" if ("THIRD PARTY DOCUMENTS ARE ACCEPTABLE" in upper or "DOKUMEN PIHAK KETIGA DIPERKENANKAN" in upper) else "未自动识别／请核对原文",
        "租船合同提单": "允许" if "CHARTER PARTY" in upper and ("ACCEPTABLE" in upper or "DIPERKENANKAN" in upper) else "未自动识别／请核对原文",
        "单据语言": "英语" if "DOCUMENTS MUST BE ISSUED IN ENGLISH" in upper else "未自动识别／请核对原文",
        "信用证号/开证日显示": "要求显示" if ("L/C NUMBER" in upper and "ISSUING DATE" in upper) or ("NOMOR DAN TANGGAL PENERBITAN" in upper) else "未自动识别／请核对原文",
    }


def _required_documents(value: str) -> str:
    upper = value.upper()
    found: list[str] = []
    for label, phrases in {
        "INVOICE": ("INVOICE",),
        "BILL OF LADING": ("BILL OF LADING", "OCEAN BILL", "AIR WAYBILL", "AIRWAY BILL"),
        "PACKING LIST": ("PACKING LIST", "PACKING",),
        "INSURANCE": ("INSURANCE", "POLICY", "CERTIFICATE OF INSURANCE"),
        "CERTIFICATE OF ORIGIN": ("CERTIFICATE OF ORIGIN",),
    }.items():
        if any(phrase in upper for phrase in phrases):
            found.append(label)
    return ",".join(found)


def _document_type(clause: str) -> str:
    upper = clause.upper()
    for name, phrases in {
        "保险单／保险凭证": ("INSURANCE", "POLICY", "CERTIFICATE OF INSURANCE"),
        "商业发票": ("COMMERCIAL INVOICE", "INVOICE"),
        "提单／运输单据": ("BILL OF LADING", "OCEAN BILL", "AIR WAYBILL", "AIRWAY BILL"),
        "装箱单": ("PACKING LIST", "PACKING"),
        "原产地证": ("CERTIFICATE OF ORIGIN",),
        "受益人证明": ("BENEFICIARY'S CERTIFICATE", "BENEFICIARY CERTIFICATE"),
        "检验证书": ("INSPECTION CERTIFICATE", "QUALITY CERTIFICATE", "ANALYSIS CERTIFICATE", "CERTIFICATE OF ANALYSIS"),
    }.items():
        if any(phrase in upper for phrase in phrases):
            return name
    return "其他／需人工识别"


def _quantity(clause: str, noun: str) -> str:
    number_words = {
        "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5",
        "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9", "TEN": "10",
    }
    match = re.search(rf"\b(FULL SET|\d{{1,3}}|{'|'.join(number_words)})\s+{noun}\b", clause.upper())
    if not match:
        return ""
    value = match.group(1)
    if value.isdigit():
        return str(int(value))
    return number_words.get(value, value)


def _original_quantity(clause: str) -> str:
    full_set = re.search(r"\bFULL\s*(?:\((\d+)\s*/\s*(\d+)\))?\s*SET\s+OF\s+ORIGINAL", clause.upper())
    if full_set:
        return full_set.group(1) or "FULL SET"
    return _quantity(clause, r"ORIGINALS?")


def parse_document_requirements(value: str) -> list[dict]:
    """Split common numbered 46A clauses into reviewable document requirements."""
    if not value.strip():
        return []
    # MT700 exports are inconsistent: both "10. CERTIFICATE" and
    # "10.CERTIFICATE" occur in field 46A.
    parts = re.split(r"(?=(?:^|\s)\d{1,2}[.)]\s*(?=[A-Z0-9]))", value.strip())
    requirements = []
    for part in parts:
        clause = re.sub(r"^\s*\d{1,2}[.)]\s*", "", part).strip()
        if not clause:
            continue
        issuer = ""
        issuer_match = re.search(r"\b(?:ISSUED|SIGNED|CERTIFIED|PREPARED|MADE OUT|DRAWN)\s+BY\s+(.+?)(?=\s+(?:IN|ON|WITH)\s+\d+\s+(?:ORIGINAL|COPY)|[.;]|$)", clause, re.I)
        if issuer_match:
            issuer = issuer_match.group(1).strip()
        requirements.append({
            "单据类型": _document_type(clause),
            "正本": _original_quantity(clause),
            "副本": _quantity(clause, r"(?:NON-NEGOTIABLE\s+)?COP(?:Y|IES)"),
            "签发／签署要求": issuer or "需人工复核原文",
            "原始条款": clause,
        })
    return requirements


def parse_mt700(message: str) -> dict:
    """Parse common MT700/MT701 fields from a text export.

    Returns ``record`` for the existing rules engine, field rows for user
    review, and non-blocking warnings when a required field is absent.
    """
    matches = list(FIELD_START.finditer(message.replace("\r", "")))
    fields: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(message)
        fields.setdefault(match.group(1), []).append(_clean(message[match.end():end]))

    def field(tag: str) -> str:
        return "\n".join(part for part in fields.get(tag, []) if part)

    currency, amount = _amount_32b(field("32B"))
    expiry = _date_from_yymmdd(field("31D"))
    latest_shipment = _date_from_yymmdd(field("44C"))
    document_requirements = parse_document_requirements(field("46A"))
    presentation_days = _presentation_days(field("48"), field("47A"))
    goods_terms = _key_goods_terms(field("45A"))
    operational_conditions = _operational_conditions(field("47A"))
    manual_review_clauses = [
        {"SWIFT 字段": tag, "条款类型": name, "完整原文": field(tag)}
        for tag, name in (("47A", "附加条款"), ("78", "对付款／承兑／议付行的指示"))
        if field(tag)
    ]
    record = {
        "报文序号": field("27"),
        "信用证编号": field("20"),
        "交易编号": field("20"),
        "信用证开证日期": _date_from_yymmdd(field("31C")),
        "信用证形式": field("40A"),
        "适用规则": field("40E"),
        "开证行": field("51A"),
        "信用证币种": currency,
        "信用证金额": amount,
        "金额容差比例": _tolerance_39a(field("39A")),
        "受益人": field("59"),
        "信用证申请人": field("50"),
        "信用证货物描述": field("45A"),
        "要求单据": _required_documents(field("46A")),
        "单据要求明细": document_requirements,
        "信用证有效期": expiry,
        "信用证到期地点": field("31D")[6:].strip(),
        "最迟装运日": latest_shipment,
        "信用证装货港": field("44E") or field("44A"),
        "信用证卸货港": field("44F") or field("44B"),
        "信用证可用方式": field("41A") or field("41D"),
        "信用证可用银行": field("41A") or field("41D"),
        "汇票期限": field("42C"),
        "汇票付款人": field("42A"),
        "分批装运": field("43P"),
        "转运": field("43T"),
        "交单期限": field("48"),
        "交单期限天数": presentation_days,
        "交单期限来源": "字段 48" if _presentation_days(field("48"), "") else ("字段 47A" if presentation_days else "未自动识别"),
        "保兑指示": field("49"),
        "费用承担": field("71D"),
        "货物关键条款": goods_terms,
        "操作性条款": operational_conditions,
        "信用证条款": field("47A"),
        "人工复核条款": manual_review_clauses,
        "MT700原文": message.strip(),
    }
    display_names = {
        "27": "报文序号", "20": "信用证编号", "31C": "开证日期", "31D": "到期日/地点", "32B": "金额", "39A": "金额容差",
        "40A": "信用证形式", "40E": "适用规则", "41A": "可用方式", "41D": "可用银行及可用方式",
        "42A": "汇票付款人", "42C": "汇票期限", "43P": "分批装运", "43T": "转运", "44C": "最迟装运日",
        "44E": "装货港", "44F": "卸货港", "45A": "货物描述", "46A": "要求单据",
        "47A": "附加条款", "48": "交单期限", "49": "保兑指示", "50": "申请人", "51A": "开证行", "59": "受益人",
        "71D": "费用承担", "78": "对付款／承兑／议付行的指示", "72Z": "发送行给接收行的信息",
    }
    field_rows = [
        {"SWIFT 字段": tag, "含义": display_names.get(tag, "未在首版映射"), "原文内容": "\n".join(values)}
        for tag, values in fields.items()
    ]
    warnings = []
    if not fields:
        warnings.append("未识别到 SWIFT 字段。请粘贴保留 :20:、:32B: 等标签的 MT700/MT701 文本。")
    if not field("20"):
        warnings.append("未识别字段 20（信用证编号）；请核对报文是否完整。")
    if not field("32B"):
        warnings.append("未识别字段 32B（信用证金额）；不会自动生成金额核验基准。")
    if not field("46A"):
        warnings.append("未识别字段 46A（要求单据）；请在手工审单页补充信用证要求。")
    return {
        "record": record,
        "fields": field_rows,
        "document_requirements": document_requirements,
        "manual_review_clauses": manual_review_clauses,
        "warnings": warnings,
    }
