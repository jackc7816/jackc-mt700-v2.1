"""Minimal Tencent Cloud OCR client for V2 image ingestion.

Credentials are read from environment variables populated by Streamlit Secrets;
they are never stored in source code or review records.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


HOST = "ocr.tencentcloudapi.com"
SERVICE = "ocr"
VERSION = "2018-11-19"
ACTION = "GeneralAccurateOCR"
# A complete documentary credit normally contains several multi-page
# documents.  Keep a conservative per-cache budget, but make it deploy-time
# configurable so previous legitimate case work cannot make a new case
# permanently unusable.  Operators can lower this in Streamlit secrets.
MAX_CLOUD_REQUESTS = int(os.getenv("TENCENT_OCR_MAX_CLOUD_REQUESTS", "20"))
CACHE_DIR = Path(os.getenv("TRADE_FINANCE_OCR_CACHE_DIR", Path(__file__).with_name(".ocr_cache")))


class TencentOCRUnavailable(RuntimeError):
    """Raised when Tencent OCR cannot be used; callers may use local fallback."""


def configured() -> bool:
    return bool(os.getenv("TENCENT_SECRET_ID", "").strip() and os.getenv("TENCENT_SECRET_KEY", "").strip())


def _cache_path(content: bytes) -> Path:
    """Use the image digest so a Streamlit rerun cannot repeat a cloud call."""
    return CACHE_DIR / f"{hashlib.sha256(content).hexdigest()}.json"


def _read_cache(content: bytes) -> dict | None:
    try:
        path = _cache_path(content)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(content: bytes, payload: dict) -> None:
    """Persist successes and failures; neither is retried automatically."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(content).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _request_count() -> int:
    """Count successful cloud responses retained in the local cache.

    A transport/authentication failure does not consume an OCR result and must
    not permanently prevent a later retry of a different document.  Failure
    entries remain cached as diagnostics, but only responses that actually
    contain a Tencent API payload are included in the budget guard.
    """
    try:
        return sum(
            1
            for path in CACHE_DIR.glob("*.json")
            if json.loads(path.read_text(encoding="utf-8")).get("response")
        )
    except OSError:
        return 0


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _authorization(payload: str, timestamp: int, secret_id: str, secret_key: str) -> str:
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    canonical_headers = f"content-type:application/json; charset=utf-8\nhost:{HOST}\n"
    signed_headers = "content-type;host"
    canonical_request = "\n".join([
        "POST", "/", "", canonical_headers, signed_headers,
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    ])
    credential_scope = f"{date}/{SERVICE}/tc3_request"
    string_to_sign = "\n".join([
        "TC3-HMAC-SHA256", str(timestamp), credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    secret_date = _hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = _hmac_sha256(secret_date, SERVICE)
    secret_signing = _hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"TC3-HMAC-SHA256 Credential={secret_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"


def parse_text_detections(response: dict) -> tuple[str, list[dict]]:
    """Convert Tencent's OCR response into V2's provider-neutral evidence form."""
    result = response.get("Response", {}) if isinstance(response, dict) else {}
    if result.get("Error"):
        raise TencentOCRUnavailable(str(result["Error"].get("Code", "Tencent OCR error")))
    detections = result.get("TextDetections", [])
    lines = []
    for index, item in enumerate(detections if isinstance(detections, list) else [], start=1):
        content = str(item.get("DetectedText", "")).strip()
        if not content:
            continue
        polygon = item.get("ItemPolygon") or {}
        confidence = item.get("Confidence", "")
        lines.append({
            "page": 1,
            "source": "tencent_ocr",
            "location": "第 1 页第 {index} 行，x={x}，y={y}".format(index=index, x=polygon.get("X", "?"), y=polygon.get("Y", "?")),
            "text": content,
            "confidence": confidence,
            "x": polygon.get("X", 0),
            "y": polygon.get("Y", 0),
            "w": polygon.get("Width", 0),
            "h": polygon.get("Height", 0),
        })
    return "\n".join(line["text"] for line in lines), lines


def recognize_image(content: bytes, filename: str) -> tuple[str, list[dict]]:
    """Call Tencent GeneralAccurateOCR for JPG/PNG images."""
    if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise TencentOCRUnavailable("腾讯云高精度 OCR 当前仅用于图片；PDF 将使用文字层/本地后备读取。")
    cached = _read_cache(content)
    if cached:
        if cached.get("response"):
            return parse_text_detections(cached["response"])
        # Failure entries from earlier versions must not block a fresh request.
        # Only a successful response is reusable evidence.
    if not configured():
        raise TencentOCRUnavailable("未配置腾讯云 OCR 密钥")
    if _request_count() >= MAX_CLOUD_REQUESTS:
        raise TencentOCRUnavailable(f"已达到腾讯云 OCR 硬上限（{MAX_CLOUD_REQUESTS} 次），未发起请求")
    secret_id = os.environ["TENCENT_SECRET_ID"].strip()
    secret_key = os.environ["TENCENT_SECRET_KEY"].strip()
    timestamp = int(time.time())
    payload = json.dumps({"ImageBase64": base64.b64encode(content).decode("ascii")}, separators=(",", ":"))
    request = Request(
        f"https://{HOST}",
        data=payload.encode("utf-8"),
        method="POST",
        headers={
            "Authorization": _authorization(payload, timestamp, secret_id, secret_key),
            "Content-Type": "application/json; charset=utf-8",
            "Host": HOST,
            "X-TC-Action": ACTION,
            "X-TC-Version": VERSION,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Region": os.getenv("TENCENT_OCR_REGION", "ap-guangzhou"),
        },
    )
    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
            _write_cache(content, {"request_made": True, "response": result})
            return parse_text_detections(result)
    except TencentOCRUnavailable:
        raise
    except Exception as exc:
        _write_cache(content, {"request_made": True, "error": f"腾讯云 OCR 调用失败：{type(exc).__name__}"})
        raise TencentOCRUnavailable(f"腾讯云 OCR 调用失败：{type(exc).__name__}") from exc
