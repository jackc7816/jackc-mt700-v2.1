"""跨境贸易融资审核助手（教学演示版）。"""
from __future__ import annotations
from datetime import date
from io import StringIO
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from core import (DISCREPANCY, NEED_INFO, NOT_APPLICABLE, PASS, WARNING, active_financing, add_financing, conclusion, evaluate, init_db, list_reviews, parsed_date, review_detail, save_review, soft_delete_review)
from mt700 import parse_mt700
from document_processing import DOCUMENT_TYPES, classify_document, extract_document, fields_to_record

st.set_page_config(page_title="信用证单据智能审核系统 V2", page_icon=":material/fact_check:", layout="wide")
init_db()

# V2 is introduced beside the preserved V1 code below.  Keeping the legacy
# implementation in this file during the transition makes rollback possible
# without touching historical SQLite snapshots.
from v2_app import run as run_v2
run_v2()
st.stop()

CASE_FILE = Path(__file__).with_name("test_cases.json")
st.session_state.setdefault("mt700_baseline", {})
st.session_state.setdefault("submission_documents", {})
st.session_state.setdefault("submission_missing", [])

def to_record(row):
    data = dict(row); aliases = {"币种":"信用证币种", "合同金额":"信用证金额", "出口商":"发票出口商", "进口商":"发票进口商"}
    for old, new in aliases.items():
        if not data.get(new) and data.get(old) is not None: data[new] = data[old]
    return {key: ("" if pd.isna(value) else value) for key, value in data.items()}

def findings_frame(findings):
    return pd.DataFrame([{ "规则":x["rule_id"], "审核项目":x["rule_name"], "状态":x["status"], "风险等级":x["severity"], "观察值":x["observed_value"], "预期值":x["expected_value"], "说明":x["explanation"], "处理建议":x["suggested_action"], "需人工复核":"是" if x["human_review_required"] else "否", "外部核验":"是" if x["external_verification"] else "否"} for x in findings])

def report(record, findings):
    triggered = [x for x in findings if x["status"] not in {PASS, NOT_APPLICABLE}]
    lines = ["# 跨境贸易融资审核报告", "", f"交易编号：{record.get('交易编号', '未提供')}", f"系统结论：**{conclusion(findings)}**", "", "## 审核发现"]
    if not triggered: lines.append("已检查的适用规则未发现异常。该结论不证明贸易真实性或完成外部核验。")
    for x in triggered:
        lines += [f"### {x['rule_id']}｜{x['rule_name']}｜{x['status']} / {x['severity']}", f"- 证据：{x['observed_value'] or '未提供'}；期望：{x['expected_value'] or '—'}", f"- 依据：{x['basis']}", f"- 说明：{x['explanation']}", f"- 建议：{x['suggested_action'] or '无需额外操作'}"]
    lines += ["", "## 系统边界", "本报告仅为教学原型的确定性规则输出，不构成银行承诺、授信决定、法律意见或制裁/承运人/市场价格核验。标记为“尚未核验”或“资料不足”的项目须由具备权限的人员通过可信外部来源完成。"]
    return "\n".join(lines)

def show_result(record, findings, key_prefix):
    st.subheader(f"本次审核：{conclusion(findings)}")
    for col, label, status in zip(st.columns(4), ["不符点", "风险预警", "资料不足", "通过"], [DISCREPANCY, WARNING, NEED_INFO, PASS]): col.metric(label, sum(x["status"] == status for x in findings))
    frame = findings_frame(findings); st.dataframe(frame, hide_index=True)
    a, b = st.columns(2)
    a.download_button("下载结构化审核结果 CSV", frame.to_csv(index=False, encoding="utf-8-sig"), "审核结果.csv", "text/csv", key=f"{key_prefix}_download_csv")
    b.download_button("下载审核报告 Markdown", report(record, findings), "审核报告.md", "text/markdown", key=f"{key_prefix}_download_report")


def _baseline_text(baseline, field, fallback=""):
    return str(baseline.get(field) or fallback)


def _baseline_number(baseline, field, fallback=0.0):
    try:
        return float(baseline.get(field) or fallback)
    except (TypeError, ValueError):
        return fallback


def _baseline_date(baseline, field):
    try:
        return date.fromisoformat(_baseline_text(baseline, field))
    except ValueError:
        return date.today()


def _apply_mt700_baseline(record):
    st.session_state["mt700_baseline"] = record
    st.session_state["submission_documents"] = {}
    st.session_state["submission_ready"] = False
    st.session_state["submission_missing"] = []
    st.session_state["review_mode"] = "单据提交中心"


def mt700_parser():
    st.subheader("MT700 / MT701 解析")
    st.caption("支持 SWIFT 纯文本报文。解析值须由审单人员核对原文；本功能不验证报文真伪，也不处理扫描件 OCR。")
    upload = st.file_uploader("上传 MT700 / MT701 文本文件", type=["txt", "mt700"], key="mt700_upload")
    pasted = st.text_area("或粘贴 MT700 / MT701 原文", height=240, key="mt700_text", placeholder=":20:信用证编号\n:32B:USD100000,00\n:46A:...\n:59:受益人")
    message = upload.getvalue().decode("utf-8", errors="replace") if upload else (pasted or "")
    if not message.strip():
        st.info("请上传或粘贴包含 :20:、:32B:、:46A: 等字段标签的 MT700 / MT701 文本。")
        return
    parsed = parse_mt700(message)
    for warning in parsed["warnings"]:
        st.warning(warning)
    expandable_tags = {"45A", "46A", "47A", "78", "72Z"}
    summary_rows = [row for row in parsed["fields"] if row["SWIFT 字段"] not in expandable_tags]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, alt="已解析的 MT700 基础字段")

    st.subheader("需展开核对的信用证条款")
    for row in parsed["fields"]:
        if row["SWIFT 字段"] not in expandable_tags:
            continue
        with st.expander(f"{row['SWIFT 字段']}｜{row['含义']}", expanded=False):
            st.code(row["原文内容"], language=None, wrap_lines=True)

    st.subheader("46A 要求单据拆分")
    if parsed["document_requirements"]:
        st.table(
            pd.DataFrame(parsed["document_requirements"]),
            hide_index=True,
            alt="根据 MT700 字段 46A 拆分的单据要求",
        )
    else:
        st.info("未能从 46A 自动识别标准单据名称；请在展开的 46A 原文中人工核对。")

    if parsed["manual_review_clauses"]:
        st.subheader("需人工复核条款")
        st.warning("47A 与 78 可能包含非标准、操作性或需结合业务背景判断的条款，系统仅保留与提示，不自动判定相符。")
        for clause in parsed["manual_review_clauses"]:
            with st.expander(f"{clause['SWIFT 字段']}｜{clause['条款类型']}（需人工复核）", expanded=False):
                st.code(clause["完整原文"], language=None, wrap_lines=True)
    st.caption("解析确认后进入单据提交中心：按 46A 要求逐份上传、确认抽取字段，再带入手工审单。")
    st.button(
        "进入单据提交中心",
        type="primary",
        key="use_mt700_baseline",
        on_click=_apply_mt700_baseline,
        args=(parsed["record"],),
    )


def render_mt700_information_panel(baseline):
    """Show extracted LC terms beside the manual input flow without overwriting them."""
    if not baseline or not baseline.get("MT700原文"):
        return
    st.subheader("信用证条款信息栏")
    st.caption("以下内容来自 MT700/MT701 自动解析，仅作手工审单基准。请以原始报文及其修改书为准。")
    key_fields = [
        ("信用证编号", "信用证编号"), ("开证日期", "信用证开证日期"), ("信用证形式", "信用证形式"),
        ("适用规则", "适用规则"), ("开证行", "开证行"), ("可用银行及方式", "信用证可用银行"),
        ("到期日", "信用证有效期"), ("到期地点", "信用证到期地点"), ("币种", "信用证币种"),
        ("金额", "信用证金额"), ("金额容差 (%)", "金额容差比例"), ("受益人", "受益人"),
        ("申请人", "信用证申请人"), ("最迟装运日", "最迟装运日"), ("装货港", "信用证装货港"),
        ("卸货港／目的地", "信用证卸货港"), ("分批装运", "分批装运"), ("转运", "转运"),
        ("交单期限", "交单期限"), ("可计算交单天数", "交单期限天数"), ("期限来源", "交单期限来源"),
        ("汇票期限", "汇票期限"), ("汇票付款人", "汇票付款人"), ("保兑指示", "保兑指示"), ("费用承担", "费用承担"),
    ]
    rows = [{"条款项目": label, "解析值": str(baseline.get(key) or "未提供／未识别")} for label, key in key_fields]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", alt="带入手工审单的信用证关键条款")
    with st.expander("货物、合同与操作性条款", expanded=False):
        terms = baseline.get("货物关键条款", {})
        operations = baseline.get("操作性条款", {})
        display = [{"类别": "货物／合同", "条款": k, "解析值": v or "未自动识别／请核对原文"} for k, v in terms.items()]
        display += [{"类别": "附加条款", "条款": k, "解析值": v} for k, v in operations.items()]
        st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch", alt="货物和操作性信用证条款")
    with st.expander("46A 单据要求明细", expanded=False):
        requirements = baseline.get("单据要求明细", [])
        if requirements:
            st.dataframe(pd.DataFrame(requirements), hide_index=True, width="stretch", alt="信用证要求单据明细")
        else:
            st.info("未能自动拆分 46A；请返回 MT700 解析页核对原文。")
    with st.expander("47A / 78 非标准及操作性条款", expanded=False):
        clauses = baseline.get("人工复核条款", [])
        if clauses:
            for clause in clauses:
                st.markdown(f"**{clause['SWIFT 字段']}｜{clause['条款类型']}**")
                st.code(clause["完整原文"], language=None, wrap_lines=True)
        else:
            st.info("报文未提供 47A 或 78。")


def _requirements_with_ids(baseline):
    return [dict(requirement, requirement_id=f"REQ-{index:02d}") for index, requirement in enumerate(baseline.get("单据要求明细", []), start=1)]


def _missing_requirements(requirements):
    documents = st.session_state["submission_documents"]
    missing = []
    for requirement in requirements:
        saved = documents.get(requirement["requirement_id"], {})
        if not saved.get("confirmed"):
            suffix = "（已上传但尚未确认）" if saved.get("filename") else ""
            missing.append(f"{requirement['单据类型']}：{requirement['原始条款'][:80]}{suffix}")
    return missing


@st.dialog("单据尚未提交")
def show_missing_documents(missing):
    st.error("尚未完成以下信用证要求单据，请重新上传或确认抽取结果：")
    for item in missing:
        st.markdown(f"- {item}")
    if st.button("返回提交中心", type="primary", key="dismiss_missing_documents"):
        st.session_state["submission_missing"] = []
        st.rerun()


def _complete_submission():
    baseline = st.session_state.get("mt700_baseline", {})
    requirements = _requirements_with_ids(baseline)
    missing = _missing_requirements(requirements)
    if missing:
        st.session_state["submission_missing"] = missing
        return
    record = dict(baseline)
    for document in st.session_state["submission_documents"].values():
        record.update(fields_to_record(document["fields"], document["document_type"]))
    st.session_state["mt700_baseline"] = record
    st.session_state["submission_ready"] = True
    st.session_state["submission_missing"] = []
    st.session_state["review_mode"] = "手工录入"


def submission_center():
    baseline = st.session_state.get("mt700_baseline", {})
    requirements = _requirements_with_ids(baseline)
    st.subheader("单据提交中心")
    st.caption("清单来自 MT700 字段 46A。每一项单据须上传并确认抽取字段后，才能带入手工审单。")
    if not requirements:
        st.warning("当前没有可用的 46A 单据清单。请先在 MT700 解析页上传完整报文。")
        return
    overview = []
    for requirement in requirements:
        saved = st.session_state["submission_documents"].get(requirement["requirement_id"], {})
        overview.append({"序号": requirement["requirement_id"], "信用证要求": requirement["单据类型"], "正本": requirement["正本"] or "未识别", "副本": requirement["副本"] or "未识别", "状态": "已确认" if saved.get("confirmed") else ("待确认" if saved.get("filename") else "未提交")})
    st.dataframe(pd.DataFrame(overview), hide_index=True, width="stretch", alt="信用证要求单据提交状态")
    st.info("本地读取 PDF 文字层，扫描件和图片使用 Windows OCR。请对照原件核对识别值并补填空字段。")

    for requirement in requirements:
        requirement_id = requirement["requirement_id"]
        saved = st.session_state["submission_documents"].get(requirement_id, {})
        title = f"{requirement_id}｜{requirement['单据类型']}"
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(requirement["原始条款"])
            uploaded = st.file_uploader("拖拽或选择此项单据", type=["pdf", "png", "jpg", "jpeg", "txt", "csv"], key=f"upload_{requirement_id}")
            if uploaded:
                content = uploaded.getvalue()
                signature = hashlib.sha256(content).hexdigest()
                if saved.get("signature") != signature or saved.get("filename") != uploaded.name:
                    classification = classify_document(uploaded.name, content)
                    with st.spinner("正在读取单据…"):
                        extraction = extract_document(uploaded.name, content, classification["document_type"])
                    saved = {"filename": uploaded.name, "signature": signature, "classification": classification, "document_type": classification["document_type"], "fields": extraction, "confirmed": False}
                    st.session_state["submission_documents"][requirement_id] = saved
                    st.session_state.pop(f"type_{requirement_id}", None)
                    st.session_state.pop(f"fields_{requirement_id}", None)
                st.caption(f"文件：{saved['filename']}｜分类来源：{saved['classification']['source']}｜置信信息：{saved['classification']['confidence']}")
                default_index = DOCUMENT_TYPES.index(saved["document_type"]) if saved["document_type"] in DOCUMENT_TYPES else len(DOCUMENT_TYPES) - 1
                confirmed_type = st.selectbox("确认单据类型", DOCUMENT_TYPES, index=default_index, key=f"type_{requirement_id}")
                if confirmed_type != saved["document_type"]:
                    saved["document_type"] = confirmed_type
                    with st.spinner("正在重新提取字段…"):
                        saved["fields"] = extract_document(saved["filename"], content, confirmed_type)
                    saved["confirmed"] = False
                    st.session_state["submission_documents"][requirement_id] = saved
                    st.session_state.pop(f"fields_{requirement_id}", None)
                fields = saved["fields"]
                st.caption(fields.get("提取状态", ""))
                if fields.get("可读文本预览"):
                    with st.expander("查看识别原文"):
                        st.text(fields["可读文本预览"])
                editable = {key: value for key, value in fields.items() if key not in {"文件名", "文档类型", "提取状态", "可读文本预览"}}
                field_rows = pd.DataFrame(list(editable.items()), columns=["字段", "提取值"])
                edited = st.data_editor(field_rows, hide_index=True, disabled=["字段"], key=f"fields_{requirement_id}", width="stretch", alt=f"{title} 的可修正抽取字段")
                if st.button("确认此单据及字段", key=f"confirm_{requirement_id}", icon=":material/task_alt:"):
                    saved["fields"].update(zip(edited["字段"], edited["提取值"].fillna("")))
                    saved["document_type"] = confirmed_type
                    saved["confirmed"] = True
                    st.session_state["submission_documents"][requirement_id] = saved
                    st.success("已确认并保存。")
            elif saved.get("filename"):
                st.info(f"已保存：{saved['filename']}。如需替换，请重新上传同一项单据。")
    st.button("开始交叉审核", type="primary", icon=":material/fact_check:", on_click=_complete_submission, key="complete_submission")
    if st.session_state.get("submission_missing"):
        show_missing_documents(st.session_state["submission_missing"])


def manual_record(baseline=None):
    baseline = baseline or {}
    from_submission = st.session_state.get("submission_ready", False)
    def doc_value(field, example=""):
        return _baseline_text(baseline, field, "" if from_submission else example)

    def doc_date(field):
        return parsed_date(baseline.get(field)) if from_submission else date.today()

    st.caption("填写信用证、发票和提单关键字段。外部核验项目不会被系统伪装为已完成。")
    render_mt700_information_panel(baseline)
    with st.form("manual_record"):
        a,b,c=st.columns(3)
        with a:
            txn=st.text_input("交易编号",_baseline_text(baseline,"交易编号","TXN-001")); lc_currency=st.selectbox("信用证币种",["USD","EUR","CNY","GBP","JPY"],index=["USD","EUR","CNY","GBP","JPY"].index(_baseline_text(baseline,"信用证币种","USD")) if _baseline_text(baseline,"信用证币种","USD") in ["USD","EUR","CNY","GBP","JPY"] else 0); lc_amount=st.number_input("信用证金额",min_value=0.0,value=_baseline_number(baseline,"信用证金额",100000.0),step=100.0); beneficiary=st.text_input("受益人",_baseline_text(baseline,"受益人","Shanghai Global Machinery Co., Ltd.")); applicant=st.text_input("信用证申请人",_baseline_text(baseline,"信用证申请人","Tokyo Industrial Trading Co., Ltd.")); goods=st.text_input("信用证货物描述",_baseline_text(baseline,"信用证货物描述","Industrial Water Pumps")); qty=st.text_input("发票数量",doc_value("发票数量", "100")); latest=st.date_input("最迟装运日",value=_baseline_date(baseline,"最迟装运日")); expiry=st.date_input("信用证有效期",value=_baseline_date(baseline,"信用证有效期"))
        with b:
            invoice = st.text_input("发票号", doc_value("发票号", "INV-2026-001"))
            inv_currency = st.text_input("发票币种", doc_value("发票币种", "USD"))
            inv_amount = st.text_input("发票金额", doc_value("发票金额", "100000.00")).replace(",", "").strip()
            exporter = st.text_input("发票出口商", doc_value("发票出口商", "Shanghai Global Machinery Co., Ltd."))
            importer = st.text_input("发票进口商", doc_value("发票进口商", "Tokyo Industrial Trading Co., Ltd."))
            inv_goods = st.text_input("发票货物描述", doc_value("发票货物描述", "Industrial Water Pumps"))
            inv_date = st.date_input("发票日期", value=doc_date("发票日期"))
            present = st.date_input("交单日", value=doc_date("交单日"))
            required_docs=st.text_input("信用证要求单据",_baseline_text(baseline,"要求单据","INVOICE,BILL OF LADING")); tolerance=st.number_input("信用证金额容差 (%)",min_value=0.0,max_value=100.0,value=_baseline_number(baseline,"金额容差比例",0.0),step=1.0); presentation_days=st.number_input("交单期限（装运后天数）",min_value=1,max_value=365,value=int(_baseline_number(baseline,"交单期限天数",21)),step=1,help="以字段 48 为优先；若字段 48 缺失，系统仅在 47A 发现明确天数时带入。请以原文及修改书为准。")
        with c:
            bl = st.text_input("提单号", doc_value("提单号", "BL-2026-001"))
            shipment = st.date_input("装运日", value=doc_date("装运日"))
            bl_goods = st.text_input("提单货物描述", doc_value("提单货物描述", inv_goods))
            bl_qty = st.text_input("提单数量", doc_value("提单数量", qty))
            load=st.text_input("信用证装货港",_baseline_text(baseline,"信用证装货港","Shanghai")); bl_load=st.text_input("提单装货港",doc_value("提单装货港", "Shanghai")); discharge=st.text_input("信用证卸货港",_baseline_text(baseline,"信用证卸货港","Yokohama")); bl_discharge=st.text_input("提单卸货港",doc_value("提单卸货港", "Yokohama")); signed=st.checkbox("承运人已签署",value=not from_submission); onboard=st.checkbox("有已装船记载",value=not from_submission); carrier=st.selectbox("承运人核验状态",["未核验","已核验","核验失败"]); sanctions=st.selectbox("制裁筛查状态",["未核验","已清除","命中"])
        submitted=st.form_submit_button("核验这一笔交易",type="primary")
    if not submitted: return None
    return {**baseline, "交易编号":txn,"发票号":invoice,"信用证币种":lc_currency,"信用证金额":lc_amount,"金额容差比例":tolerance,"发票币种":inv_currency,"发票金额":inv_amount,"受益人":beneficiary,"发票出口商":exporter,"信用证申请人":applicant,"发票进口商":importer,"信用证货物描述":goods,"发票货物描述":inv_goods,"发票数量":qty,"提单号":bl,"提单货物描述":bl_goods,"提单数量":bl_qty,"最迟装运日":latest.isoformat(),"装运日":(shipment.isoformat() if shipment else ""),"信用证装货港":load,"提单装货港":bl_load,"信用证卸货港":discharge,"提单卸货港":bl_discharge,"信用证有效期":expiry.isoformat(),"交单日":(present.isoformat() if present else ""),"发票日期":(inv_date.isoformat() if inv_date else ""),"承运人签署":signed,"已装船记载":onboard,"承运人核验状态":carrier,"制裁筛查状态":sanctions,"要求单据":required_docs,"交单期限天数":presentation_days}

def history():
    st.subheader("审核历史"); st.caption("每次审核均创建独立记录。页面删除为软删除，数据库保留删除事件这一必要审计痕迹。")
    reviews=list_reviews()
    if not reviews: st.info("尚无审核记录。"); return
    for item in reviews:
        with st.expander(f"{item['created_at'][:19]} · {item['transaction_id'] or '未命名'} · {item['conclusion']}"):
            run, findings=review_detail(item["id"]); st.caption(f"记录 ID：{item['id']} ｜规则版本：{item['rule_version']} ｜来源：{item['source']}")
            payload=json.loads(run["payload"])
            if item["rule_version"] == "legacy-json":
                st.info("此记录保留自旧版 JSON 历史。为避免用新版规则重写旧结论，以下展示的是原始快照。")
                st.dataframe(pd.DataFrame(payload.get("legacy_results", [])),use_container_width=True,hide_index=True)
                st.caption("原始录入快照"); st.dataframe(pd.DataFrame(payload.get("legacy_records", [])),use_container_width=True,hide_index=True)
            else: show_result(payload, findings, f"history_{item['id']}")
            if st.button("从界面删除此审核记录",key=f"delete_{item['id']}"): soft_delete_review(item["id"]); st.rerun()

st.title("🔎 AI 跨境贸易融资审核助手")
st.caption("教学演示：UCP 600/ISBP 审单规则与融资风险预警分层；系统不替代人工银行审核。")
tab_review,tab_cases,tab_finance,tab_history=st.tabs(["录入与审核","10组模拟案例","融资台账","审核历史"])
with tab_review:
    mode=st.segmented_control("录入方式",["手工录入","CSV 批量上传","MT700 解析","单据提交中心"],default="手工录入",key="review_mode",width="stretch"); record=None
    if mode=="手工录入": record=manual_record(st.session_state.get("mt700_baseline"))
    elif mode=="CSV 批量上传":
        upload=st.file_uploader("选择 CSV",type="csv")
        if upload:
            df=pd.read_csv(upload); st.dataframe(df,use_container_width=True)
            if st.button("核验当前 CSV",type="primary"):
                for _,row in df.iterrows():
                    data=to_record(row); save_review(data,evaluate(data,active_financing()),"CSV 批量上传")
                st.success(f"已创建 {len(df)} 条审核记录。请在“审核历史”查看。")
    elif mode=="MT700 解析":
        mt700_parser()
    else:
        submission_center()
    if record:
        findings=evaluate(record,active_financing()); save_review(record,findings,"手工录入"); show_result(record, findings, "manual_current")
with tab_cases:
    cases=json.loads(CASE_FILE.read_text(encoding="utf-8")); selected=st.selectbox("选择教学模拟案例",[x["id"]+"｜"+x["name"] for x in cases]); case=next(x for x in cases if selected.startswith(x["id"])); st.write(case["note"]); st.json(case["record"])
    if st.button("运行此案例",type="primary"):
        findings=evaluate(case["record"],active_financing()); save_review(case["record"],findings,case["id"]); show_result(case["record"], findings, f"case_{case['id']}")
with tab_finance:
    st.caption("登记既有、未结清融资，以供 RISK001 在本行台账范围内进行重复融资提示；系统不会检索其他银行记录。")
    with st.form("finance"):
        invoice_no=st.text_input("发票号"); exporter=st.text_input("出口商"); tx=st.text_input("交易编号"); amount=st.number_input("融资金额",min_value=0.0); outstanding=st.number_input("未结清余额",min_value=0.0); saved=st.form_submit_button("登记融资台账")
    if saved: add_financing(invoice_no,exporter,tx,amount,outstanding); st.success("已登记融资台账记录。")
    st.dataframe(pd.DataFrame(active_financing()),use_container_width=True,hide_index=True)
with tab_history: history()
