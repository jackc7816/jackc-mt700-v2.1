"""Deterministic, evidence-backed V2 documentary review.

Rules here deliberately handle only values with stated evidence.  Everything
else is returned as a human-review item rather than being guessed as a
discrepancy.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import re


PASS = "PASS"
DISCREPANCY = "DISCREPANCY"
NEED_INFO = "NEED_INFO"
NOT_APPLICABLE = "NOT_APPLICABLE"


def _normal(value: object) -> str:
    return " ".join(str(value or "").upper().split())


def _number(value: object) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def _date(value: object) -> date | None:
    raw = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _finding(rule_id: str, name: str, status: str, dimension: str, basis: str, observed: object = "", expected: object = "", evidence: list[dict] | None = None, explanation: str = "") -> dict:
    return {
        "rule_id": rule_id,
        "rule_name": name,
        "status": status,
        "risk_dimension": dimension,
        "basis": basis,
        "observed_value": str(observed or ""),
        "expected_value": str(expected or ""),
        "evidence": evidence or [],
        "explanation": explanation,
        "human_review_required": status in {DISCREPANCY, NEED_INFO},
    }


def _document_by_type(documents: list[dict], document_type: str) -> dict | None:
    return next((doc for doc in documents if doc.get("document_type") == document_type), None)


def _structured_value(document: dict | None, name: str) -> str:
    if not document:
        return ""
    item = document.get("structured", {}).get("fields", {}).get(name, {})
    return str(item.get("value", "")) if isinstance(item, dict) else ""


def _field_evidence(document: dict | None, field: str) -> list[dict]:
    if not document:
        return []
    value = str(document.get("fields", {}).get(field, "")).strip()
    if not value:
        return []
    matches = [item for item in document.get("evidence", []) if value.upper() in str(item.get("text", "")).upper()]
    return matches[:3] or [{"source": "人工确认字段", "location": document.get("filename", ""), "text": value}]


def _compare(rule_id: str, name: str, dimension: str, basis: str, actual: object, expected: object, evidence: list[dict]) -> dict:
    if not _normal(expected):
        return _finding(rule_id, name, NOT_APPLICABLE, dimension, basis, explanation="信用证未识别出可比较要求。")
    if not _normal(actual):
        return _finding(rule_id, name, NEED_INFO, dimension, basis, expected=expected, evidence=evidence, explanation="缺少可核对的单据字段；需人工确认。")
    if _normal(actual) != _normal(expected):
        return _finding(rule_id, name, DISCREPANCY, dimension, basis, actual, expected, evidence, "单据值与信用证要求不一致。")
    return _finding(rule_id, name, PASS, dimension, basis, actual, expected, evidence, "已按可用字段核对。")


def build_review(lc: dict, requirements: list[dict], documents: list[dict]) -> dict:
    findings: list[dict] = []
    for requirement in requirements:
        matching = [doc for doc in documents if doc.get("requirement_id") == requirement.get("requirement_id")]
        findings.append(
            _finding(
                f"REQ-{requirement.get('requirement_id', 'UNKNOWN')}",
                f"{requirement.get('单据类型', '单据')}提交状态",
                PASS if matching and matching[0].get("confirmed") else DISCREPANCY,
                "份数风险",
                "信用证字段 46A 的明确单据要求",
                "已确认" if matching and matching[0].get("confirmed") else "缺失或未确认",
                requirement.get("原始条款", ""),
                matching[0].get("evidence", [])[:2] if matching else [],
                "单据已提交并由审核员确认。" if matching and matching[0].get("confirmed") else "信用证要求单据尚未完成提交/确认。",
            )
        )

    invoice = _document_by_type(documents, "商业发票")
    bill = _document_by_type(documents, "提单／运输单据")
    insurance = _document_by_type(documents, "保险单／保险凭证")
    packing = _document_by_type(documents, "装箱单")
    origin = _document_by_type(documents, "原产地证")
    analysis = _document_by_type(documents, "检验证书")
    invoice_fields = invoice.get("fields", {}) if invoice else {}
    bill_fields = bill.get("fields", {}) if bill else {}

    findings.append(_compare("LC-AMOUNT", "商业发票金额", "金额风险", "信用证字段 32B 及明确金额条款", invoice_fields.get("发票金额"), lc.get("信用证金额"), _field_evidence(invoice, "发票金额")))
    findings.append(_compare("LC-CURRENCY", "商业发票币种", "金额风险", "信用证字段 32B", invoice_fields.get("发票币种"), lc.get("信用证币种"), _field_evidence(invoice, "发票币种")))
    findings.append(_compare("LC-BENEFICIARY", "发票开具人", "签署风险", "信用证字段 59 及商业发票要求", invoice_fields.get("发票出口商"), lc.get("受益人"), _field_evidence(invoice, "发票出口商")))
    findings.append(_compare("LC-LOAD-PORT", "装货港", "运输风险", "信用证字段 44E/44A", bill_fields.get("提单装货港"), lc.get("信用证装货港"), _field_evidence(bill, "提单装货港")))
    findings.append(_compare("LC-DISCHARGE-PORT", "卸货港", "运输风险", "信用证字段 44F/44B", bill_fields.get("提单卸货港"), lc.get("信用证卸货港"), _field_evidence(bill, "提单卸货港")))

    latest, shipped = _date(lc.get("最迟装运日")), _date(bill_fields.get("装运日"))
    if latest and shipped:
        findings.append(_finding("LC-SHIP-DATE", "最迟装运日", DISCREPANCY if shipped > latest else PASS, "时间风险", "信用证字段 44C", shipped, latest, _field_evidence(bill, "装运日"), "装运日晚于最迟装运日。" if shipped > latest else "装运日未晚于最迟装运日。"))
    else:
        findings.append(_finding("LC-SHIP-DATE", "最迟装运日", NEED_INFO, "时间风险", "信用证字段 44C", evidence=_field_evidence(bill, "装运日"), explanation="最迟装运日或单据装运日未能可靠取得。"))

    comparisons = [
        ("货物描述", "发票货物描述", "提单货物描述", "货物风险"),
        ("数量", "发票数量", "提单数量", "货物风险"),
    ]
    extended_comparisons = [
        ("订单编号", invoice, "", packing, "order_number", "货物风险"),
        ("装箱单与提单卸货港", packing, "port_of_discharge", bill, "port_of_discharge", "运输风险"),
        ("保险单与提单装货港", insurance, "port_of_loading", bill, "port_of_loading", "保险风险"),
        ("原产地证与发票货物", origin, "goods_description", invoice, "", "货物风险"),
        ("检验证书与装箱单净重", analysis, "total_net_weight", packing, "total_net_weight", "货物风险"),
    ]
    matrix = []
    for label, invoice_key, bill_key, dimension in comparisons:
        left, right = invoice_fields.get(invoice_key, ""), bill_fields.get(bill_key, "")
        status = NOT_APPLICABLE if not invoice and not bill else (NEED_INFO if not _normal(left) or not _normal(right) else (PASS if _normal(left) == _normal(right) else DISCREPANCY))
        matrix.append({"比较字段": label, "商业发票": left or "未取得", "运输单据": right or "未取得", "结果": status, "证据": _field_evidence(invoice, invoice_key) + _field_evidence(bill, bill_key)})
        findings.append(_finding(f"CROSS-{label}", f"跨单据{label}一致性", status, dimension, "UCP 600 第14(d)条；以信用证明确要求为先", left, right, matrix[-1]["证据"], "存在可比较字段不一致。" if status == DISCREPANCY else "资料不足，需人工确认。" if status == NEED_INFO else "已比较。"))

    for label, left_doc, left_key, right_doc, right_key, dimension in extended_comparisons:
        left = _structured_value(left_doc, left_key) if left_key else ""
        right = _structured_value(right_doc, right_key) if right_key else ""
        # Legacy invoice fields remain the source for invoice comparisons.
        if label == "原产地证与发票货物":
            right = invoice_fields.get("发票货物描述", "")
        status = NOT_APPLICABLE if not left_doc and not right_doc else (NEED_INFO if not _normal(left) or not _normal(right) else (PASS if _normal(left) == _normal(right) else DISCREPANCY))
        matrix.append({"比较字段": label, "左侧": left or "未取得", "右侧": right or "未取得", "结果": status, "证据": []})
        findings.append(_finding(f"CROSS-EXT-{label}", f"跨单据{label}一致性", status, dimension, "仅比较两份单据均载有且可确认的字段", left, right, [], "资料不足，需人工确认。" if status == NEED_INFO else "存在可比较字段不一致。" if status == DISCREPANCY else "已比较。"))

    if insurance:
        findings.append(_finding("INSURANCE-PRESENT", "保险单据可读性", NEED_INFO if not insurance.get("text") else PASS, "保险风险", "信用证字段 46A 及保险条款", evidence=insurance.get("evidence", [])[:3], explanation="已上传保险单据。" if insurance.get("text") else "保险单据未读出文字，需人工核对。"))
    else:
        findings.append(_finding("INSURANCE-PRESENT", "保险单据", NOT_APPLICABLE, "保险风险", "信用证字段 46A", explanation="未见独立保险单据要求或未上传。"))

    # Soft/non-standard clauses are not auto-discrepancies.
    for clause in lc.get("人工复核条款", []):
        findings.append(_finding(f"MANUAL-{clause.get('SWIFT 字段', 'CLAUSE')}", "非标准/操作性条款", NEED_INFO, "软条款风险", f"信用证字段 {clause.get('SWIFT 字段', '')}", expected=clause.get("完整原文", ""), evidence=[{"source": "MT700", "location": f"字段 {clause.get('SWIFT 字段', '')}", "text": clause.get("完整原文", "")}], explanation="该条款需人工解释及确认，不自动判定为不符。"))

    risks = {}
    dimensions = ["时间风险", "金额风险", "货物风险", "运输风险", "保险风险", "签署风险", "份数风险", "软条款风险"]
    for dimension in dimensions:
        states = [item["status"] for item in findings if item["risk_dimension"] == dimension]
        risks[dimension] = "RED" if DISCREPANCY in states else ("YELLOW" if NEED_INFO in states else "GREEN")
    overall = "RED" if "RED" in risks.values() else ("YELLOW" if "YELLOW" in risks.values() else "GREEN")
    conclusion = "存在确定性不符点" if overall == "RED" else ("存在待人工确认项目" if overall == "YELLOW" else "已通过可自动核验项目")
    return {"lc": lc, "requirements": requirements, "documents": documents, "findings": findings, "matrix": matrix, "risk_dimensions": risks, "summary": {"risk_level": overall, "conclusion": conclusion}}
