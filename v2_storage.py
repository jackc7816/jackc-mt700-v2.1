"""Versioned storage for V2 reviews.

The V1 tables are intentionally never altered or recomputed.  V2 review
snapshots live in their own table so the history view can safely show both
versions during the transition.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from uuid import uuid4


DB_PATH = Path(__file__).with_name("trade_finance_audit.db")
V2_RULE_VERSION = "v2.0-deterministic"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_v2_db() -> None:
    """Create additive V2 tables only; no V1 table or record is changed."""
    with _connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_review_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                lc_number TEXT,
                conclusion TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                rule_version TEXT NOT NULL,
                payload TEXT NOT NULL,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS v2_audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                detail TEXT NOT NULL
            );
            """
        )


def _event(con: sqlite3.Connection, event_type: str, entity_id: str, detail: dict) -> None:
    con.execute(
        "INSERT INTO v2_audit_events(occurred_at,event_type,entity_id,detail) VALUES(?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), event_type, entity_id, json.dumps(detail, ensure_ascii=False)),
    )


def save_v2_review(review: dict) -> str:
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    summary = review.get("summary", {})
    with _connect() as con:
        con.execute(
            "INSERT INTO v2_review_runs VALUES(?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                created_at,
                str(review.get("lc", {}).get("信用证编号", "")),
                str(summary.get("conclusion", "待人工复核")),
                str(summary.get("risk_level", "YELLOW")),
                V2_RULE_VERSION,
                json.dumps(review, ensure_ascii=False, default=str),
                0,
                None,
            ),
        )
        _event(con, "CREATE", run_id, {"rule_version": V2_RULE_VERSION})
    return run_id


def list_v2_reviews() -> list[dict]:
    with _connect() as con:
        return [dict(row) for row in con.execute(
            "SELECT * FROM v2_review_runs WHERE is_deleted=0 ORDER BY created_at DESC"
        )]


def v2_review_detail(run_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute("SELECT payload FROM v2_review_runs WHERE id=? AND is_deleted=0", (run_id,)).fetchone()
    return json.loads(row["payload"]) if row else None


def soft_delete_v2_review(run_id: str) -> None:
    with _connect() as con:
        con.execute(
            "UPDATE v2_review_runs SET is_deleted=1, deleted_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), run_id),
        )
        _event(con, "SOFT_DELETE", run_id, {})
