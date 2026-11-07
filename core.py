"""Deterministic review rules and local SQLite audit store for the teaching demo.

The module deliberately separates documentary discrepancies from financing-risk
alerts.  It does not perform carrier, sanctions, ownership or market-price
lookups: callers must explicitly provide a verification result.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

RULE_VERSION = "2026.10.30"
DB_PATH = Path(__file__).with_name("trade_finance_audit.db")
LEGACY_HISTORY_PATH = Path(__file__).with_name(".audit_history.json")

PASS, DISCREPANCY, WARNING, NEED_INFO, NOT_APPLICABLE = (
    "PASS", "DISCREPANCY", "WARNING", "NEED_INFO", "NOT_APPLICABLE"
)

RULES = {
    "LC001": ("发票币种", "UCP 600 第18条"),
    "LC002": ("发票金额及容差", "UCP 600 第18、30条及信用证条款"),
    "LC003": ("发票开具主体/受益人", "UCP 600 第18条及信用证条款"),
    "LC004": ("发票货物描述", "UCP 600 第18条"),
    "LC005": ("要求单据完整性", "UCP 600 第14条及信用证条款"),
    "LC006": ("信用证有效期", "UCP 600 第6条"),
    "BL001": ("最迟装运日期", "UCP 600 第20条及信用证条款"),
    "BL002": ("装货港", "UCP 600 第20条及信用证条款"),
    "BL003": ("卸货港", "UCP 600 第20条及信用证条款"),
    "BL004": ("承运人/签署", "UCP 600 第20条"),
    "BL005": ("已装船记载", "UCP 600 第20条"),
    "BL006": ("交单期限", "UCP 600 第14(c)条及信用证条款"),
    "DOC001": ("货物描述跨单据一致性", "UCP 600 第14(d)条"),
    "DOC002": ("数量跨单据一致性", "UCP 600 第14(d)条"),
    "DOC003": ("交易主体跨单据一致性", "UCP 600 第14(d)条"),
    "DOC004": ("日期逻辑", "UCP 600 第14(d)条；以条款为准"),
    "RISK001": ("重复融资", "融资风险预警：本行台账交叉检查"),
    "RISK002": ("运输单据真实性", "融资风险预警：需独立承运人核验"),
    "RISK003": ("异常定价", "融资风险预警：需价格依据及增强尽调"),
    "RISK004": ("制裁/交易主体风险", "融资风险预警：需有权限的名单筛查"),
}


def text(value: object) -> str:
    return " ".join(str(value or "").strip().upper().split())


def dec(value: object) -> Decimal | None:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, AttributeError):
        return None


def parsed_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (ValueError, AttributeError):
        return None


def truthy(value: object) -> bool:
    return text(value) in {"YES", "Y", "TRUE", "1", "是", "已装船", "VERIFIED", "CLEARED"}


def finding(rule_id: str, status: str, severity: str, observed: object = "", expected: object = "", explanation: str = "", action: str = "", external: bool = False) -> dict:
    name, basis = RULES[rule_id]
    return {"rule_id": rule_id, "rule_name": name, "status": status, "severity": severity,
            "observed_value": str(observed or ""), "expected_value": str(expected or ""),
            "basis": basis, "explanation": explanation, "suggested_action": action,
            "human_review_required": status in {DISCREPANCY, WARNING, NEED_INFO},
            "external_verification": external}


def compare(rule_id: str, actual: object, expected: object, label: str, severity: str = "MEDIUM") -> dict:
    if not text(expected):
        return finding(rule_id, NOT_APPLICABLE, "LOW", explanation=f"未提供信用证{label}条款，规则不适用。")
    if not text(actual):
        return finding(rule_id, NEED_INFO, severity, expected=expected, explanation=f"缺少{label}，无法核验。", action="补充相应单据字段。")
    if text(actual) != text(expected):
        return finding(rule_id, DISCREPANCY, severity, actual, expected, f"{label}与信用证/基准资料存在冲突。", "核对信用证修改、单据原件及条款。")
    return finding(rule_id, PASS, "LOW", actual, expected, f"{label}已核对，未见冲突。")


def evaluate(record: dict, active_financing: list[dict] | None = None) -> list[dict]:
    """Evaluate all 20 documented rules. Missing optional LC fields are N/A, not errors."""
    r = {str(k): v for k, v in record.items()}
    out: list[dict] = []
    # LC / invoice
    out.append(compare("LC001", r.get("发票币种"), r.get("信用证币种"), "币种", "HIGH"))
    limit, amount, tol = dec(r.get("信用证金额")), dec(r.get("发票金额")), dec(r.get("金额容差比例") or 0)
    if limit is None:
        out.append(finding("LC002", NOT_APPLICABLE, "LOW", explanation="未提供信用证金额，规则不适用。"))
    elif amount is None:
        out.append(finding("LC002", NEED_INFO, "HIGH", expected=limit, explanation="发票金额缺失或格式无效。", action="补充发票金额。"))
    elif amount > limit * (Decimal("1") + tol / Decimal("100")):
        out.append(finding("LC002", DISCREPANCY, "HIGH", amount, f"{limit}（容差 {tol}%）", "发票金额超过信用证金额及录入容差。", "核对信用证金额、容差和有效修改；不可自动认定拒付。"))
    else:
        out.append(finding("LC002", PASS, "LOW", amount, limit, "发票金额在录入的信用证金额/容差内。"))
    out.append(compare("LC003", r.get("发票出口商"), r.get("受益人"), "受益人/发票开具主体", "HIGH"))
    out.append(compare("LC004", r.get("发票货物描述"), r.get("信用证货物描述"), "货物描述", "MEDIUM"))
    required = [text(x) for x in str(r.get("要求单据") or "").replace("，", ",").split(",") if text(x)]
    supplied = {x for x, key in [("INVOICE", "发票号"), ("BILL OF LADING", "提单号"), ("INSURANCE", "保险单号"), ("PACKING LIST", "装箱单号")] if text(r.get(key))}
    if not required:
        out.append(finding("LC005", NOT_APPLICABLE, "LOW", explanation="未录入信用证要求单据清单，规则不适用。"))
    else:
        missing = [x for x in required if x not in supplied]
        out.append(finding("LC005", DISCREPANCY if missing else PASS, "HIGH" if missing else "LOW", ", ".join(supplied), ", ".join(required), "缺少信用证要求单据：" + ", ".join(missing) if missing else "要求单据已齐备。", "补齐单据或核对信用证修改。" if missing else ""))
    presentation, expiry = parsed_date(r.get("交单日")), parsed_date(r.get("信用证有效期"))
    if not expiry:
        out.append(finding("LC006", NOT_APPLICABLE, "LOW", explanation="未提供信用证有效期，规则不适用。"))
    elif not presentation:
        out.append(finding("LC006", NEED_INFO, "HIGH", expected=expiry, explanation="缺少交单日，无法核验有效期。", action="补充交单日。"))
    else:
        out.append(finding("LC006", DISCREPANCY if presentation > expiry else PASS, "HIGH" if presentation > expiry else "LOW", presentation, expiry, "交单日超过信用证有效期。" if presentation > expiry else "交单日在信用证有效期内。", "核对有效信用证修改。" if presentation > expiry else ""))
    # BL
    shipment, latest = parsed_date(r.get("装运日")), parsed_date(r.get("最迟装运日"))
    if not latest:
        out.append(finding("BL001", NOT_APPLICABLE, "LOW", explanation="未提供最迟装运日，规则不适用。"))
    elif not shipment:
        out.append(finding("BL001", NEED_INFO, "HIGH", expected=latest, explanation="缺少提单装运日。", action="补充已装船日期。"))
    else:
        out.append(finding("BL001", DISCREPANCY if shipment > latest else PASS, "HIGH" if shipment > latest else "LOW", shipment, latest, "装运日迟于最迟装运日。" if shipment > latest else "装运日未晚于最迟装运日。", "核对信用证修改。" if shipment > latest else ""))
    out.append(compare("BL002", r.get("提单装货港"), r.get("信用证装货港"), "装货港", "HIGH"))
    out.append(compare("BL003", r.get("提单卸货港"), r.get("信用证卸货港"), "卸货港", "HIGH"))
    if not text(r.get("提单号")):
        out.append(finding("BL004", NOT_APPLICABLE, "LOW", explanation="未提供提单，运输单据签署规则不适用。"))
        out.append(finding("BL005", NOT_APPLICABLE, "LOW", explanation="未提供提单，已装船记载规则不适用。"))
    else:
        signed = truthy(r.get("承运人签署")); onboard = truthy(r.get("已装船记载"))
        out.append(finding("BL004", PASS if signed else DISCREPANCY, "LOW" if signed else "HIGH", r.get("承运人签署"), "是", "提单含承运人/其代理人签署信息。" if signed else "提单缺少符合要求的承运人/代理人签署信息。", "核验正本提单签署栏。" if not signed else ""))
        out.append(finding("BL005", PASS if onboard else DISCREPANCY, "LOW" if onboard else "HIGH", r.get("已装船记载"), "是", "已装船记载已提供。" if onboard else "未见已装船记载。", "核验提单已装船批注。" if not onboard else ""))
    days = int(dec(r.get("交单期限天数") or 21) or 21)
    if not shipment or not presentation:
        out.append(finding("BL006", NEED_INFO if (text(r.get("提单号")) or text(r.get("交单日"))) else NOT_APPLICABLE, "HIGH", explanation="缺少装运日或交单日，无法核验交单期限。", action="补充日期。"))
    else:
        deadline = shipment + timedelta(days=days)
        allowed = min(deadline, expiry) if expiry else deadline
        out.append(finding("BL006", DISCREPANCY if presentation > allowed else PASS, "HIGH" if presentation > allowed else "LOW", presentation, allowed, "交单日晚于适用交单期限。" if presentation > allowed else "交单日在适用期限内。", "核对信用证另行规定的交单期限与有效期。" if presentation > allowed else ""))
    # cross-document: non-identical inputs are conflicts, but absence does not assert conflict
    out.append(compare("DOC001", r.get("提单货物描述"), r.get("发票货物描述"), "发票与提单货物描述", "MEDIUM"))
    out.append(compare("DOC002", r.get("提单数量"), r.get("发票数量"), "发票与提单数量", "MEDIUM"))
    out.append(compare("DOC003", r.get("发票进口商"), r.get("信用证申请人"), "申请人/发票进口商", "MEDIUM"))
    invoice_date = parsed_date(r.get("发票日期"))
    if not invoice_date or not shipment:
        out.append(finding("DOC004", NOT_APPLICABLE, "LOW", explanation="日期资料不足，未作日期逻辑结论。发票早于装运本身并非不符点。"))
    elif presentation and presentation < shipment:
        out.append(finding("DOC004", DISCREPANCY, "HIGH", presentation, shipment, "交单日早于装运日，日期逻辑冲突。", "核对单据日期。"))
    else:
        out.append(finding("DOC004", PASS, "LOW", explanation="未发现交单早于装运的日期逻辑冲突；发票早于装运不单独构成不符点。"))
    # Financing risks
    active_financing = active_financing or []
    applicant_exporter = r.get("发票出口商") or r.get("出口商")
    duplicate = [x for x in active_financing if text(x.get("invoice_no")) == text(r.get("发票号")) and text(x.get("exporter")) == text(applicant_exporter) and dec(x.get("outstanding_amount", 0)) and dec(x.get("outstanding_amount", 0)) > 0]
    out.append(finding("RISK001", WARNING if duplicate else PASS, "HIGH" if duplicate else "LOW", ", ".join(x.get("id", "") for x in duplicate), "无未结清同一应收账款融资", "发现本行未结清的同发票融资记录。" if duplicate else "本行台账未见相同发票的未结清融资记录。", "暂停自动通过，核查融资余额及跨行融资。" if duplicate else "", bool(duplicate)))
    carrier = text(r.get("承运人核验状态"))
    if carrier in {"VERIFIED", "已核验"}:
        out.append(finding("RISK002", PASS, "LOW", carrier, "已核验", "承运人核验结果已由录入方标记为已核验。", external=True))
    elif carrier in {"FAILED", "失败", "核验失败"}:
        out.append(finding("RISK002", WARNING, "HIGH", carrier, "已核验", "承运人独立核验失败，存在真实性重大疑点。", "人工调查并核验承运人原始记录。", True))
    else:
        out.append(finding("RISK002", NEED_INFO, "MEDIUM", carrier, "外部承运人核验", "系统未连接承运人数据源，真实性尚未核验。", "通过可信独立渠道完成核验。", True))
    reference, unit = dec(r.get("参考单价")), dec(r.get("申报单价"))
    if reference is None or unit is None:
        out.append(finding("RISK003", NEED_INFO, "LOW", explanation="缺少可验证的参考单价或申报单价，未作异常定价判断。", action="补充定价依据；系统不提供市场价格查询。", external=True))
    else:
        deviation = abs(unit - reference) / reference if reference else Decimal("0")
        out.append(finding("RISK003", WARNING if deviation > Decimal("0.30") else PASS, "MEDIUM" if deviation > Decimal("0.30") else "LOW", f"偏离 {deviation:.1%}", "不高于30%", "价格偏离参考值超过30%。" if deviation > Decimal("0.30") else "价格偏离未超过教学阈值。", "核对定价依据、合同变更和交易习惯。" if deviation > Decimal("0.30") else "", True))
    sanctions = text(r.get("制裁筛查状态"))
    if sanctions in {"CLEARED", "已清除", "已核验"}:
        out.append(finding("RISK004", PASS, "LOW", sanctions, "已完成筛查", "筛查结果由录入方标记为已清除。", external=True))
    elif sanctions in {"HIT", "命中"}:
        out.append(finding("RISK004", WARNING, "HIGH", sanctions, "未命中", "名单筛查命中，必须人工合规复核。", "暂停处理并按机构流程升级。", True))
    else:
        out.append(finding("RISK004", NEED_INFO, "MEDIUM", sanctions, "外部名单筛查", "系统未接入实时名单数据，制裁/交易主体风险尚未核验。", "使用获授权的数据源完成筛查。", True))
    return out


def conclusion(findings: list[dict]) -> str:
    statuses = {x["status"] for x in findings}
    if DISCREPANCY in statuses: return "不符点—人工复核"
    if WARNING in statuses: return "风险预警—人工复核"
    if NEED_INFO in statuses: return "资料不足待核验"
    return "已检查项目通过"


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS review_runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL, transaction_id TEXT, conclusion TEXT NOT NULL, rule_version TEXT NOT NULL, payload TEXT NOT NULL, is_deleted INTEGER NOT NULL DEFAULT 0, deleted_at TEXT);
        CREATE TABLE IF NOT EXISTS review_findings (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, rule_id TEXT NOT NULL, status TEXT NOT NULL, severity TEXT NOT NULL, finding_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS financing_records (id TEXT PRIMARY KEY, invoice_no TEXT NOT NULL, exporter TEXT NOT NULL, transaction_id TEXT, financed_amount TEXT NOT NULL, outstanding_amount TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit_events (id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL, event_type TEXT NOT NULL, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, detail TEXT NOT NULL);
        """)
        # One-time, non-destructive migration of the former JSON history.  The
        # original file remains untouched; legacy result snapshots are kept in
        # payload so an old result is never silently recomputed under new rules.
        has_runs = con.execute("SELECT 1 FROM review_runs LIMIT 1").fetchone()
        if not has_runs and LEGACY_HISTORY_PATH.exists():
            try:
                legacy = json.loads(LEGACY_HISTORY_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                legacy = []
            if isinstance(legacy, list):
                for entry in legacy:
                    if not isinstance(entry, dict):
                        continue
                    payload = {"legacy_records": entry.get("records", []), "legacy_results": entry.get("results", [])}
                    run_id = str(entry.get("id") or uuid4())
                    created = str(entry.get("reviewed_at") or datetime.now(timezone.utc).isoformat())
                    transaction = ""
                    if entry.get("records") and isinstance(entry["records"], list):
                        transaction = str(entry["records"][0].get("交易编号", ""))
                    con.execute("INSERT OR IGNORE INTO review_runs VALUES(?,?,?,?,?,?,?,?,?)", (run_id, created, "原 JSON 历史迁移", transaction, "历史快照（旧版规则）", "legacy-json", json.dumps(payload, ensure_ascii=False), 0, None))
                    _event(con, "MIGRATE", "review_run", run_id, {"source": "legacy JSON"})


def _event(con: sqlite3.Connection, event: str, entity: str, entity_id: str, detail: dict) -> None:
    con.execute("INSERT INTO audit_events(occurred_at,event_type,entity_type,entity_id,detail) VALUES(?,?,?,?,?)", (datetime.now(timezone.utc).isoformat(), event, entity, entity_id, json.dumps(detail, ensure_ascii=False)))


def active_financing() -> list[dict]:
    with connect() as con:
        return [dict(x) for x in con.execute("SELECT * FROM financing_records WHERE status='ACTIVE' AND CAST(outstanding_amount AS REAL)>0")]


def save_review(record: dict, findings: list[dict], source: str) -> str:
    run_id, now = str(uuid4()), datetime.now(timezone.utc).isoformat()
    with connect() as con:
        con.execute("INSERT INTO review_runs VALUES(?,?,?,?,?,?,?,?,?)", (run_id, now, source, str(record.get("交易编号", "")), conclusion(findings), RULE_VERSION, json.dumps(record, ensure_ascii=False, default=str), 0, None))
        con.executemany("INSERT INTO review_findings(run_id,rule_id,status,severity,finding_json) VALUES(?,?,?,?,?)", [(run_id, x["rule_id"], x["status"], x["severity"], json.dumps(x, ensure_ascii=False)) for x in findings])
        _event(con, "CREATE", "review_run", run_id, {"source": source, "rule_version": RULE_VERSION})
    return run_id


def list_reviews() -> list[dict]:
    with connect() as con:
        return [dict(x) for x in con.execute("SELECT * FROM review_runs WHERE is_deleted=0 ORDER BY created_at DESC")]


def review_detail(run_id: str) -> tuple[dict, list[dict]]:
    with connect() as con:
        run = dict(con.execute("SELECT * FROM review_runs WHERE id=?", (run_id,)).fetchone())
        findings = [json.loads(x["finding_json"]) for x in con.execute("SELECT finding_json FROM review_findings WHERE run_id=? ORDER BY id", (run_id,))]
    return run, findings


def soft_delete_review(run_id: str) -> None:
    with connect() as con:
        con.execute("UPDATE review_runs SET is_deleted=1, deleted_at=? WHERE id=?", (datetime.now(timezone.utc).isoformat(), run_id))
        _event(con, "SOFT_DELETE", "review_run", run_id, {})


def add_financing(invoice_no: str, exporter: str, transaction_id: str, amount: object, outstanding: object) -> str:
    finance_id = str(uuid4())
    with connect() as con:
        con.execute("INSERT INTO financing_records VALUES(?,?,?,?,?,?,?,?)", (finance_id, invoice_no, exporter, transaction_id, str(amount), str(outstanding), "ACTIVE", datetime.now(timezone.utc).isoformat()))
        _event(con, "CREATE", "financing_record", finance_id, {"invoice_no": invoice_no})
    return finance_id
