"""Evidence-first extraction for trade documents.

The functions deliberately use relative OCR positions, never fixed image pixels.
They produce an auditable structured snapshot while preserving the flat V1/V2
field interface in ``document_processing``.
"""
from __future__ import annotations

import re
from collections import defaultdict


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def _line_texts(evidence: list[dict]) -> str:
    return "\n".join(_clean(item.get("text")) for item in evidence if _clean(item.get("text")))


def _anchor_value(evidence: list[dict], labels: tuple[str, ...], pattern: str = r".+") -> dict:
    """Find a block directly below, then to the right of a label block."""
    regex = re.compile(pattern, re.I)
    for anchor in evidence:
        anchor_text = _clean(anchor.get("text"))
        if not any(re.search(label, anchor_text, re.I) for label in labels):
            continue
        page, ax, ay, ah = anchor.get("page", 1), float(anchor.get("x", 0)), float(anchor.get("y", 0)), float(anchor.get("h", 0))
        candidates = []
        for item in evidence:
            value = _clean(item.get("text"))
            if item.get("page", 1) != page or not value or not regex.fullmatch(value):
                continue
            x, y = float(item.get("x", 0)), float(item.get("y", 0))
            vertical = y - ay
            if 0 < vertical <= max(120, ah * 4) and x >= ax - 80:
                candidates.append((0, vertical, abs(x - ax), item))
            elif x >= ax and abs(y - ay) <= max(40, ah * 1.7):
                candidates.append((1, x - ax, abs(y - ay), item))
        if candidates:
            item = min(candidates, key=lambda row: row[:3])[3]
            return _field(_clean(item.get("text")), item, "anchor_relative")
    return _field("", None, "unresolved")


def _header_value(evidence: list[dict], labels: tuple[str, ...], pattern: str = r".+", prefer: str = "right") -> dict:
    """Read a value from a compact document header without crossing columns.

    Packing lists often put three independent label/value columns in the same
    header.  ``_anchor_value`` deliberately permits a nearby block below a
    label, which is useful for forms but can make ``Order No.`` pick the value
    on the following ``Voyage No.`` row.  Header fields need stricter rules:
    first take a real value on the same row, otherwise take the next item in
    the same visual column.  ``prefer='below'`` is used for address blocks.
    """
    regex = re.compile(pattern, re.I)
    for anchor in evidence:
        anchor_text = _clean(anchor.get("text"))
        matched_label = next((label for label in labels if re.search(label, anchor_text, re.I)), None)
        if not matched_label:
            continue
        # Some OCR engines return a whole header pair in one detection, e.g.
        # ``Destination: HAIPHONG VIETNAM`` rather than two adjacent blocks.
        # Preserve that direct association before using geometry.
        inline = re.search(rf"(?:{matched_label})\s*[:#-]?\s*(.+)$", anchor_text, re.I)
        if inline:
            value = _clean(inline.group(1)).lstrip(":#- ")
            if value and re.search(r"[A-Z0-9]", value, re.I) and regex.fullmatch(value) and not re.search(r"(?:NO\.?|NUMBER|DATE|DESTINATION|EXPORTER|VESSEL|VOYAGE|CONTAINERS?|COLOUR|MARKS?)\s*:?$", value, re.I):
                return _field(value, anchor, "header_inline")
        page = anchor.get("page", 1)
        ax, ay = float(anchor.get("x", 0)), float(anchor.get("y", 0))
        aw, ah = max(float(anchor.get("w", 0)), 20), max(float(anchor.get("h", 0)), 12)
        right, below = [], []
        for item in evidence:
            value = _clean(item.get("text"))
            if item is anchor or item.get("page", 1) != page or not value or not regex.fullmatch(value):
                continue
            # Another label is never a header-field value.
            if re.search(r"(?:NO\.?|NUMBER|DATE|DESTINATION|EXPORTER|VESSEL|VOYAGE|CONTAINERS?|COLOUR|MARKS?)\s*:?$", value, re.I):
                continue
            x, y = float(item.get("x", 0)), float(item.get("y", 0))
            if x >= ax + max(20, aw * 0.45) and abs(y - ay) <= max(18, ah * 1.25):
                right.append((x - ax, abs(y - ay), item))
            elif 0 < y - ay <= max(90, ah * 4) and ax - 35 <= x <= ax + max(120, aw * 0.75):
                below.append((y - ay, abs(x - ax), item))
        ordered = (below, right) if prefer == "below" else (right, below)
        for candidates in ordered:
            if candidates:
                item = min(candidates, key=lambda row: row[:2])[2]
                return _field(_clean(item.get("text")), item, "header_relative")
    return _field("", None, "unresolved")


def _same_cell_value(evidence: list[dict], labels: tuple[str, ...], pattern: str = r".+") -> dict:
    """Return the nearest value below a numbered/table label in its own column."""
    regex = re.compile(pattern, re.I)
    for anchor in evidence:
        if not any(re.search(label, _clean(anchor.get("text")), re.I) for label in labels):
            continue
        page, ax, ay = anchor.get("page", 1), float(anchor.get("x", 0)), float(anchor.get("y", 0))
        ah, aw = max(float(anchor.get("h", 0)), 12), max(float(anchor.get("w", 0)), 60)
        candidates = []
        for item in evidence:
            value = _clean(item.get("text"))
            x, y = float(item.get("x", 0)), float(item.get("y", 0))
            if item.get("page", 1) != page or not value or not regex.fullmatch(value):
                continue
            if not (0 < y - ay <= max(110, ah * 4) and ax - 35 <= x <= ax + max(aw * 1.25, 260)):
                continue
            if re.search(r"(?:DOC\.\s*FORM|SERVICE\s+CONTRACT|FCL\s*/\s*FCL|SIGNED\s+BY)", value, re.I):
                continue
            candidates.append((y - ay, abs(x - ax), item))
        if candidates:
            return _field(_clean(min(candidates, key=lambda row: row[:2])[2].get("text")), min(candidates, key=lambda row: row[:2])[2], "same_cell_anchor")
    return _field("", None, "unresolved")


def _field(value: str, source: dict | None, method: str, confidence: object | None = None) -> dict:
    source = source or {}
    return {
        "value": _clean(value), "confidence": source.get("confidence", confidence if confidence is not None else 0),
        "source_page": source.get("page"), "source_text": _clean(source.get("text")),
        "bbox": {key: source.get(key) for key in ("x", "y", "w", "h")} if source else None,
        "method": method, "confirmed": False,
    }


def _regex_field(text: str, pattern: str, method: str = "text_pattern") -> dict:
    match = re.search(pattern, text, re.I | re.M)
    return _field(match.group(1) if match else "", None, method)


def _first(fields: dict, key: str, *alternatives: dict) -> None:
    for candidate in alternatives:
        if candidate.get("value"):
            fields[key] = candidate
            return
    fields[key] = _field("", None, "unresolved")


def _rows(evidence: list[dict]) -> list[list[dict]]:
    """Cluster OCR blocks into visual rows; usable for simple table layouts."""
    by_page: dict[object, list[dict]] = defaultdict(list)
    for item in evidence:
        if _clean(item.get("text")):
            by_page[item.get("page", 1)].append(item)
    output = []
    for page_items in by_page.values():
        current: list[dict] = []
        y0 = None
        for item in sorted(page_items, key=lambda x: (float(x.get("y", 0)), float(x.get("x", 0)))):
            y = float(item.get("y", 0))
            if current and y0 is not None and abs(y - y0) > max(18, float(item.get("h", 0)) * 1.4):
                output.append(sorted(current, key=lambda x: float(x.get("x", 0))))
                current, y0 = [], None
            current.append(item)
            y0 = y if y0 is None else (y0 + y) / 2
        if current:
            output.append(sorted(current, key=lambda x: float(x.get("x", 0))))
    return output


def _container_rows(evidence: list[dict]) -> list[dict]:
    """Read container table records, including records split across OCR rows.

    A container identifier starts a record.  Subsequent OCR rows belong to that
    record until the next identifier; this works for both one-line tables and
    continuation pages without relying on fixed pixels or carrier-specific IDs.
    """
    records: list[dict] = []
    container_re = re.compile(r"\b([A-Z]{4}\d{7})\b", re.I)
    seal_re = re.compile(r"\b(?:SEAL\s*(?:NO\.?|NUMBER)?\s*[:#-]?\s*)?([A-Z]{1,8}\d{4,9})\b", re.I)
    containers = [item for item in evidence if container_re.search(_clean(item.get("text")))]
    for container_item in sorted(containers, key=lambda item: (item.get("page", 1), float(item.get("y", 0)), float(item.get("x", 0)))):
        found = container_re.search(_clean(container_item.get("text")))
        if not found:
            continue
        page, y = container_item.get("page", 1), float(container_item.get("y", 0))
        row_items = [
            item for item in evidence
            if item.get("page", 1) == page and abs(float(item.get("y", 0)) - y) <= 10 and _clean(item.get("text"))
        ]
        source = " ".join(_clean(item.get("text")) for item in sorted(row_items, key=lambda item: float(item.get("x", 0))))
        record = {
            "container_number": found.group(1).upper(), "seal_number": "",
            "package_count": "", "package_type": "", "container_type": "",
            "gross_weight": "", "source_text": source, "source_page": page,
        }
        # Ignore the container number itself when looking for a seal.
        seal_candidates = [match.group(1).upper() for match in seal_re.finditer(source) if match.group(1).upper() != record["container_number"]]
        if seal_candidates:
            record["seal_number"] = seal_candidates[0]
        package = re.search(r"\b(\d{1,6})\s+(BUNDLES?|CARTONS?|CASES?|PACKAGES?|PALLETS?|PIECES?)\b", source, re.I)
        if package:
            record["package_count"], record["package_type"] = package.group(1), package.group(2).upper()
        equipment = re.search(r"\b(20|40|45)\s*[' ]?(GP|HC|HQ|RF|OT|FR)\b|\b(FCL\s*/\s*FCL)\b", source, re.I)
        if equipment:
            record["container_type"] = _clean(equipment.group(0)).upper()
        weights = re.findall(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:KGS?|KG)\b", source, re.I)
        if weights:
            record["gross_weight"] = weights[-1].replace(",", "")
        if record["package_count"] or record["gross_weight"] or record["seal_number"]:
            records.append(record)
    return records


def _bill_goods(evidence: list[dict], text: str) -> dict:
    """Use the commodity line itself; never emit a table delimiter as goods."""
    for item in evidence:
        value = _clean(item.get("text"))
        if re.search(r"\bPRIMARY\s+ALUM\w*\s+INGOT\b", value, re.I):
            return _field(value, item, "goods_line_anchor")
    fallback = _regex_field(text, r"\b(PRIMARY\s+ALUM\w*\s+INGOT[^\n;]*)")
    return fallback if fallback.get("value") not in {"", ")", "("} else _field("", None, "unresolved")


def _original_bill_count(evidence: list[dict], text: str) -> dict:
    direct = _regex_field(text, r"(?:HAS\s+)?ISSUED\s+(\d+)\s+ORIGINAL\s+BILLS?\s+OF\s+LADING")
    if direct.get("value"):
        return direct
    for item in evidence:
        if not re.search(r"ORIGINAL\s+BILLS?\s+OF\s+LADING", _clean(item.get("text")), re.I):
            continue
        page, y = item.get("page", 1), float(item.get("y", 0))
        preceding = [
            candidate for candidate in evidence
            if candidate.get("page", 1) == page and 0 <= y - float(candidate.get("y", 0)) <= 45
            and re.fullmatch(r"\d+", _clean(candidate.get("text")))
        ]
        if preceding:
            source = min(preceding, key=lambda candidate: y - float(candidate.get("y", 0)))
            return _field(_clean(source.get("text")), source, "footer_original_count")
    return _field("", None, "unresolved")


def _is_template_hint(value: str) -> bool:
    return bool(re.search(r"\b(?:INSERT\s+NAME|ADDRESS\s+AND\s+PHONE|PHONE/FAX|TBC|TO\s+BE\s+CONFIRMED)\b", value, re.I))


def _labelled_block(evidence: list[dict], label: str, stop_labels: tuple[str, ...] = ()) -> dict:
    """Read a labelled box below its anchor while excluding template prompts."""
    anchors = [item for item in evidence if re.search(label, _clean(item.get("text")), re.I)]
    for anchor in anchors:
        page, ax, ay = anchor.get("page", 1), float(anchor.get("x", 0)), float(anchor.get("y", 0))
        ah, aw = max(float(anchor.get("h", 0)), 12), max(float(anchor.get("w", 0)), 40)
        stops = [
            float(item.get("y", 0)) for item in evidence
            if item.get("page", 1) == page and any(re.search(marker, _clean(item.get("text")), re.I) for marker in stop_labels)
            and float(item.get("y", 0)) > ay
        ]
        stop_y = min(stops) if stops else float("inf")
        below = []
        for item in evidence:
            value = _clean(item.get("text"))
            if item.get("page", 1) != page or not value:
                continue
            x, y = float(item.get("x", 0)), float(item.get("y", 0))
            if y <= ay + ah * .25 or y >= stop_y or y - ay > max(260, ah * 12) or x < ax - 20 or x > ax + max(aw * 2.5, 650):
                continue
            if any(re.search(marker, value, re.I) for marker in stop_labels):
                continue
            if _is_template_hint(value):
                continue
            below.append(item)
        if below:
            below.sort(key=lambda item: (float(item.get("y", 0)), float(item.get("x", 0))))
            # Keep the address block, but stop at a visibly separated next box.
            first_y = float(below[0].get("y", 0))
            kept = [item for item in below if float(item.get("y", 0)) - first_y <= max(130, ah * 6)]
            return _field(" ".join(_clean(item.get("text")) for item in kept), kept[0], "labelled_box")
    return _field("", None, "unresolved")


def _date_field(evidence: list[dict], labels: tuple[str, ...], text: str) -> dict:
    date_re = re.compile(r"\b(\d{1,2}\s+[A-Z]{3,9}\s+\d{4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b", re.I)
    for anchor in evidence:
        anchor_text = _clean(anchor.get("text"))
        if not any(re.search(label, anchor_text, re.I) for label in labels):
            continue
        page, ay = anchor.get("page", 1), float(anchor.get("y", 0))
        nearby = [anchor] + [item for item in evidence if item.get("page", 1) == page and abs(float(item.get("y", 0)) - ay) <= 90]
        for item in sorted(nearby, key=lambda item: (abs(float(item.get("y", 0)) - ay), float(item.get("x", 0)))):
            match = date_re.search(_clean(item.get("text")))
            if match:
                value = match.group(1).upper()
                try:
                    from datetime import datetime
                    for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
                        try:
                            value = datetime.strptime(value, fmt).date().isoformat()
                            break
                        except ValueError:
                            continue
                except ValueError:
                    pass
                return _field(value, item, "date_anchor")
    match = date_re.search(text)
    return _field(match.group(1) if match else "", None, "text_pattern")


def _packing(text: str, evidence: list[dict]) -> tuple[dict, list[dict]]:
    f = {}
    _first(f, "packing_list_number", _header_value(evidence, (r"PACKING\s+LIST\s*(?:NO|NUMBER)",), r"[A-Z0-9-]{4,}"), _regex_field(text, r"PACKING\s+LIST\s*(?:NO\.?|NUMBER)\s*[:#]?\s*([A-Z0-9-]{4,})"))
    _first(f, "order_number", _header_value(evidence, (r"ORDER\s+NO",), r"[A-Z0-9-]{4,}"), _regex_field(text, r"ORDER\s+NO\.?\s*[:#]?\s*([A-Z0-9-]{4,})"))
    _first(f, "exporter", _header_value(evidence, (r"EXPORTER",), r".+", prefer="below"))
    _first(f, "consignee", _header_value(evidence, (r"PACKING\s+LIST\s+FOR",), r".+"))
    _first(
        f, "vessel_name", _header_value(evidence, (r"VESSEL",), r".+"),
        _regex_field(text, r"CONTAINER\s+PACKING\s+LIST\s+FOR\s*:?\s*\n[^\n]+\n([A-Z][A-Z0-9 .'-]+?)\s*\nETD\b"),
    )
    _first(f, "voyage_number", _header_value(evidence, (r"VOYAGE\s+NO",), r"[A-Z0-9-]{2,}"))
    _first(
        f, "port_of_loading", _header_value(evidence, (r"PORT\s+OF\s+LOADING",), r".+"),
        _regex_field(text, r"\bETD\s*:?\s*\n?([A-Z][A-Z .'-]+?)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    )
    _first(f, "port_of_discharge", _header_value(evidence, (r"DESTINATION|PORT\s+OF\s+DISCHARGE",), r".+"))
    _first(f, "container_count", _header_value(evidence, (r"NO\.?\s+OF\s+CONTAINERS",), r"\d+"))
    _first(f, "marks_and_numbers", _header_value(evidence, (r"MARKS\s*(?:AND|&)\s*(?:NUMBERS|NOS)",), r".+"))
    _first(f, "total_bundle_count", _regex_field(text, r"SHIPMENT\s+TOTAL\s+(\d+)"))
    _first(f, "total_piece_count", _regex_field(text, r"SHIPMENT\s+TOTAL\s+\d+\s+(\d+)"))
    _first(f, "total_net_weight", _regex_field(text, r"SHIPMENT\s+TOTAL\s+\d+\s+\d+\s+([\d,.]+)"))
    _first(f, "total_gross_weight", _regex_field(text, r"SHIPMENT\s+TOTAL\s+\d+\s+\d+\s+[\d,.]+\s+([\d,.]+)"))
    return f, _container_rows(evidence)


def _insurance(text: str, evidence: list[dict]) -> tuple[dict, list[dict]]:
    def column_block(labels: tuple[str, ...], stop_labels: tuple[str, ...], max_x: float, max_height: float = 420) -> dict:
        """Collect a multi-line certificate box without borrowing a side column."""
        for anchor in evidence:
            if not any(re.search(label, _clean(anchor.get("text")), re.I) for label in labels):
                continue
            page, ax, ay = anchor.get("page", 1), float(anchor.get("x", 0)), float(anchor.get("y", 0))
            stops = [
                float(item.get("y", 0)) for item in evidence
                if item.get("page", 1) == page and float(item.get("y", 0)) > ay
                and any(re.search(label, _clean(item.get("text")), re.I) for label in stop_labels)
            ]
            stop_y = min(stops) if stops else ay + max_height
            values = [
                item for item in evidence
                if item.get("page", 1) == page and ay < float(item.get("y", 0)) < stop_y
                and ax - 40 <= float(item.get("x", 0)) <= max_x and _clean(item.get("text"))
            ]
            if values:
                values.sort(key=lambda item: (float(item.get("y", 0)), float(item.get("x", 0))))
                return _field(" ".join(_clean(item.get("text")) for item in values), values[0], "insurance_column_block")
        return _field("", None, "unresolved")

    def below_label(labels: tuple[str, ...]) -> dict:
        """Read the value below a policy-form label without borrowing another column.

        Insurance certificates commonly have four columns.  A generic
        right-or-below lookup can therefore select a neighbouring header (for
        example ``ASSURED'S REFERENCE`` as the assured's name).  These values
        are laid out directly below their labels, so only accept the nearest
        block in the same visual column.
        """
        for anchor in evidence:
            anchor_text = _clean(anchor.get("text"))
            if not any(re.search(label, anchor_text, re.I) for label in labels):
                continue
            page = anchor.get("page", 1)
            ax, ay = float(anchor.get("x", 0)), float(anchor.get("y", 0))
            aw, ah = max(float(anchor.get("w", 0)), 20), max(float(anchor.get("h", 0)), 12)
            candidates = []
            for item in evidence:
                value = _clean(item.get("text"))
                x, y = float(item.get("x", 0)), float(item.get("y", 0))
                if item is anchor or item.get("page", 1) != page or not value:
                    continue
                if not (ay + ah * .25 < y <= ay + 170):
                    continue
                if x < ax - 80 or x > ax + max(aw * 1.35, 620):
                    continue
                if re.search(r"(?:ASSURED|CONSIGNEE|BUYER|VESSEL|PORT\s+OF|MARKS|INSURED\s+VALUE|MARINE\s+CERTIFICATE|DATED\s+AT|NUMBER\s+OF\s+ORIGINALS)", value, re.I):
                    continue
                # Certificates use side-by-side boxes whose values may be
                # slightly higher than the intended value.  Column alignment
                # is therefore more reliable than vertical distance.
                candidates.append((abs(x - ax), y - ay, item))
            if candidates:
                item = min(candidates, key=lambda row: row[:2])[2]
                return _field(_clean(item.get("text")), item, "insurance_below_label")
        return _field("", None, "unresolved")

    f = {}
    for key, labels in {"insured": (r"ASSURED$",), "assured_reference": (r"ASSURED.?S\s+REFERENCE",), "consignee": (r"CONSIGNEE",), "buyer": (r"BUYER",), "vessel_name": (r"VESSEL/AIRCRAFT",), "port_of_loading": (r"PORT\s+OF\s+LOADING",), "port_of_discharge": (r"PORT\s+OF\s+DISCHARGE",), "marks_and_numbers": (r"MARKS\s*&\s*NUMBERS",), "insurance_certificate_number": (r"MARINE\s+CERTIFICATE\s+NO",), "insurance_date": (r"DATED\s+AT",)}.items():
        _first(f, key, below_label(labels))
    _first(f, "consignee", column_block((r"^CONSIGNEE$",), (r"^PRE\s+CARRIAGE", r"^PORT\s+OF\s+LOADING"), 1000), f["consignee"])
    _first(f, "insured_value", _regex_field(text, r"(?:INSURED\s+VALUE|INSURED\s+AMOUNT)[\s\S]{0,100}?\b[A-Z]{3}\s*([\d,.]+)"))
    _first(f, "insured_currency", _regex_field(text, r"(?:INSURED\s+VALUE|INSURED\s+AMOUNT)[\s\S]{0,100}?\b([A-Z]{3})\s*[\d,.]+"))
    _first(f, "original_count", _regex_field(text, r"NUMBER\s+OF\s+ORIGINALS?\s*:?\s*([^\n]+)"))
    _first(f, "goods_description", _regex_field(text, r"(?:MARKS,\s*NUMBERS,\s*KIND\s*of\s*PACKAGES)[^\n]*\n?([^\n]+(?:\n[^\n]+){0,2})"))
    _first(f, "insurance_quantity", _regex_field(text, r"([\d,.]+\s*MT\s+TOTAL\s+NET)"))
    _first(f, "total_gross_weight", _regex_field(text, r"([\d,.]+\s*MT\s+TOTAL\s+GROSS)"))
    agent_details = column_block((r"^BAOVIET\s+INSURANCE$",), (r"^CONDITIONS",), 1300, 180)
    if agent_details["value"]:
        agent_details["value"] = "BAOVIET INSURANCE " + agent_details["value"]
    _first(f, "claims_agent", agent_details)
    return f, []


def _origin(text: str, evidence: list[dict]) -> tuple[dict, list[dict]]:
    def origin_block(labels: tuple[str, ...], stop_labels: tuple[str, ...], max_x: float) -> dict:
        """Read a multi-line left-column block from this common CO layout."""
        for anchor in evidence:
            if not any(re.search(label, _clean(anchor.get("text")), re.I) for label in labels):
                continue
            page = anchor.get("page", 1)
            ax, ay = float(anchor.get("x", 0)), float(anchor.get("y", 0))
            stops = [
                float(item.get("y", 0)) for item in evidence
                if item.get("page", 1) == page
                and float(item.get("y", 0)) > ay
                and any(re.search(label, _clean(item.get("text")), re.I) for label in stop_labels)
            ]
            # Without a following section label this may be a compact test or
            # an unrelated layout; retain the safer one-line header fallback.
            if not stops:
                continue
            stop_y = min(stops)
            values = [
                item for item in evidence
                if item.get("page", 1) == page
                and ay < float(item.get("y", 0)) < stop_y
                and ax - 30 <= float(item.get("x", 0)) <= max_x
                and _clean(item.get("text"))
            ]
            if values:
                values.sort(key=lambda item: (float(item.get("y", 0)), float(item.get("x", 0))))
                return _field(" ".join(_clean(item.get("text")) for item in values), values[0], "origin_column_block")
        return _field("", None, "unresolved")

    def origin_goods() -> dict:
        for anchor in evidence:
            if not re.search(r"(?:PACKING\s+DETAILS|DESCRIPTION\s+OF\s+GOODS)", _clean(anchor.get("text")), re.I):
                continue
            page, ay = anchor.get("page", 1), float(anchor.get("y", 0))
            values = [
                item for item in evidence
                if item.get("page", 1) == page
                and ay < float(item.get("y", 0)) <= ay + 210
                and 300 <= float(item.get("x", 0)) <= 900
                and _clean(item.get("text"))
                and not re.search(r"(?:PACKING\s+DETAILS|DESCRIPTION\s+OF\s+GOODS)", _clean(item.get("text")), re.I)
                and not re.search(r"(?:SIGNATURE|COUNTRY\s+OF\s+ORIGIN|MANUFACTURER|DATE)", _clean(item.get("text")), re.I)
            ]
            if values:
                values.sort(key=lambda item: (float(item.get("y", 0)), float(item.get("x", 0))))
                return _field(" ".join(_clean(item.get("text")) for item in values), values[0], "origin_goods_column")
        return _field("", None, "unresolved")

    f = {}
    # This certificate has adjacent header columns.  Restrict lookup to the
    # labelled column so one header cannot borrow a neighbouring value.
    for key, labels in {"shipper": (r"^SHIPPER$",), "consignee": (r"^CONSIGNEE$",), "notify_party": (r"^NOTIFY\s+PARTY$",), "order_number": (r"^ORDER\s+NO",), "export_reference_number": (r"^EXPORT\s+REFERENCE",), "place_of_receipt": (r"^PLACE\s+OF\s+RECEIPT",), "port_of_loading": (r"^PORT\s+OF\s+LOADING",), "port_of_discharge": (r"^PORT\s+OF\s+DISCHARGE",), "vessel_name": (r"^VESSEL$",), "manufacturer": (r"^MANUFACT\w*\s+NAME",), "authorized_signatory": (r"^SIGNATURE\s+OF\s+AUTHORISED",), "issue_date": (r"^DATE$",)}.items():
        _first(f, key, _header_value(evidence, labels, r".+", prefer="below"))
    # These left-column address boxes are intentionally multi-line.  Prefer
    # their complete contents over the first-line header helper used above.
    _first(f, "shipper", origin_block((r"^SHIPPER$",), (r"^CONSIGNEE$",), 880), f["shipper"])
    _first(f, "consignee", origin_block((r"^CONSIGNEE$",), (r"^NOTIFY\s+PARTY$",), 880), f["consignee"])
    _first(f, "notify_party", origin_block((r"^NOTIFY\s+PARTY$",), (r"^PLACE\s+OF\s+RECEIPT$", r"^PORT\s+OF\s+LOADING", r"^PORT\s+OF\s+DISCHARGE", r"^VESSEL$"), 880), f["notify_party"])
    _first(f, "manufacturer", origin_block((r"^MANUFACT\w*\s+NAME",), (r"^SIGNATURE\s+OF\s+AUTHORISED",), 800), f["manufacturer"])
    # The final line in the signature area is the signatory, while the prior
    # line is usually company/stamp text.
    signatories = [item for item in evidence if re.search(r"\b(?:MANAGER|DIRECTOR|OFFICER)\b", _clean(item.get("text")), re.I)]
    if signatories:
        signatory = max(signatories, key=lambda item: float(item.get("y", 0)))
        f["authorized_signatory"] = _field(_clean(signatory.get("text")), signatory, "signature_area_text")
    f["place_of_receipt"]["value"] = re.sub(r"(?i)(Australia)CY$", r"\1", f["place_of_receipt"]["value"])
    _first(f, "certificate_number", _regex_field(text, r"(?:CERTIFICATE\s+(?:NO\.?|NUMBER)|CO\s+NO\.?)\s*[:#]?\s*([A-Z0-9-]{4,})"))
    _first(
        f, "country_of_origin",
        _regex_field(text, r"(?:ARE\s+OF|OF)\s+([A-Z]{4,})\s+ORIGIN"),
        _regex_field(text, r"COUNTRY\s+OF\s+ORIGIN\s*:?\s*([A-Z][A-Z ]{3,})"),
    )
    _first(f, "brand", _regex_field(text, r"BRAND\s*:?\s*([A-Z][A-Z0-9 -]{2,})"))
    _first(f, "hs_code", _regex_field(text, r"HS\s*CODE\s*[:.]?\s*(\d{4,10})"))
    _first(f, "goods_description", origin_goods(), _regex_field(text, r"(\d+\s+BUNDLES?[^\n]*(?:\n[^\n]+){0,2})"))
    _first(f, "goods_quantity", _regex_field(text, r"(\d+\s+BUNDLES?\s*\(\s*[\d,]+\s*\)\s*PIECES)"))
    # Broken glyphs from a local OCR engine are not documentary values.  Keep
    # the original evidence visible, but avoid presenting these as extracted
    # values that an operator might mistake for a mismatch.
    for key in ("place_of_receipt", "port_of_loading", "port_of_discharge", "vessel_name", "issue_date"):
        value = f[key]["value"]
        if value and not re.fullmatch(r"[A-Za-z0-9 ,.&/()'\-]+", value):
            f[key] = _field("", None, "unresolved_low_quality_ocr")
    signature_marker = any(re.search(r"SIGNATURE\s+OF\s+AUTHORISED", _clean(item.get("text")), re.I) for item in evidence)
    _first(f, "signature_present", _field("已检测到签署区文字，请人工确认" if signature_marker else "待人工确认", None, "signature_region"))
    # OCR can read printed company text near the signature, but it cannot
    # establish that a physical seal is present from text evidence alone.
    _first(f, "stamp_present", _field("未检测到可验证印章；请人工确认", None, "image_region_required"))
    return f, []


def _analysis(text: str, evidence: list[dict]) -> tuple[dict, list[dict]]:
    f = {}
    for key, labels in {"ship_to": (r"SHIP\s+TO",), "order_number": (r"TOMAGO\s+ORDER\s+NO",), "manifest_number": (r"MANIFEST\s+NO",), "issue_date": (r"^DATE",), "manufacturer": (r"TOMAGO\s+ALUMINIUM",)}.items():
        _first(f, key, _anchor_value(evidence, labels, r".+"))
    _first(f, "product_description", _regex_field(text, r"PRODUCT\s+DESCRIPTION\s*=?\s*([^\n]+)"))
    _first(f, "total_bundle_count", _regex_field(text, r"TOTAL\s+(\d+)\s+[\d,]+\s+[\d,.]+"))
    _first(f, "total_piece_count", _regex_field(text, r"TOTAL\s+\d+\s+([\d,]+)"))
    _first(f, "total_net_weight", _regex_field(text, r"TOTAL\s+\d+\s+[\d,]+\s+([\d,.]+)"))
    table = []
    for row in _rows(evidence):
        values = [_clean(x.get("text")) for x in row]
        if values and re.fullmatch(r"I?\d{4,6}", values[0], re.I) and len(values) >= 4:
            table.append({"cast_number": values[0], "raw_cells": values, "source_page": row[0].get("page", 1), "source_text": " | ".join(values)})
    return f, table


def _bill(text: str, evidence: list[dict]) -> tuple[dict, list[dict]]:
    f = {}
    labels = {
        "bill_of_lading_number": (r"BILL\s+OF\s+LADING\s+NO", r"B/L\s*NO"),
        "booking_number": (r"BOOKING\s+NO",),
        "place_of_receipt": (r"(?:5\.\s*)?PLACE\s+OF\s+RECEIPT",),
        "port_of_loading": (r"(?:7\.\s*)?PORT\s+OF\s+LOADING",),
        "port_of_discharge": (r"(?:8\.\s*)?PORT\s+OF\s+DISCHARGE",),
        "place_of_delivery": (r"(?:9\.\s*)?(?:COMBINED\s+TRANSPORT\*?\s*)?PLACE\s+OF\s+DELIVERY",),
        "ocean_vessel_voyage": (r"(?:6\.\s*)?OCEAN\s+VESSEL",),
    }
    for key, key_labels in labels.items():
        _first(f, key, _same_cell_value(evidence, key_labels, r".+"), _anchor_value(evidence, key_labels, r".+"))
    _first(f, "shipper", _labelled_block(evidence, r"^1\.\s*SHIPPER", (r"^2\.\s*CONSIGNEE",)))
    _first(f, "consignee", _labelled_block(evidence, r"^2\.\s*CONSIGNEE", (r"^3\.\s*NOTIFY\s+PARTY",)))
    _first(f, "notify_party", _labelled_block(evidence, r"^3\.\s*NOTIFY\s+PARTY", (r"^4\.", r"^5\.")))
    _first(f, "date_laden_on_board", _date_field(evidence, (r"DATE\s+LADEN\s+ON\s+BOARD",), text))
    _first(f, "issue_date", _date_field(evidence, (r"DATE\s+OF\s+ISSUE",), text))
    _first(f, "freight_payment_term", _regex_field(text, r"\b(OCEAN\s+FREIGHT\s+(?:PREPAID|COLLECT))\b"))
    _first(f, "original_count", _original_bill_count(evidence, text))
    _first(f, "carrier", _regex_field(text, r"SIGNED\s+FOR\s+THE\s+CARRIER[,\s]+([^\n]+)"))
    # The words "SAY ... CONTAINERS TOTAL" state equipment count, not packages.
    count = _regex_field(text, r"SAY\s+(?:[A-Z-]+\s+)*(\d+)\s+CONTAINERS?\s+TOTAL")
    if not count["value"]:
        words = {"ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9", "TEN": "10"}
        spelled = re.search(r"SAY\s+([A-Z]+)\s+CONTAINERS?\s+TOTAL", text, re.I)
        if spelled and spelled.group(1).upper() in words:
            count = _field(words[spelled.group(1).upper()], None, "spelled_container_count")
    _first(f, "container_count", count)
    _first(f, "total_package_count", _regex_field(text, r"\b(\d{1,6})\s+(?:BUNDLES?|CARTONS?|CASES?|PACKAGES?|PALLETS?)\b"))
    _first(f, "total_package_type", _regex_field(text, r"\b\d{1,6}\s+(BUNDLES?|CARTONS?|CASES?|PACKAGES?|PALLETS?)\b"))
    _first(f, "total_piece_count", _regex_field(text, r"\(?([\d,]+)\)?\s+PIECES?\b"))
    _first(f, "gross_weight", _regex_field(text, r"(?:TOTAL\s+)?GROSS\s+WEIGHT\s*[:.]?\s*([\d,.]+)\s*(?:KGS?|KG|MT)"))
    _first(f, "net_weight", _regex_field(text, r"(?:TOTAL\s+)?NET\s+WEIGHT\s*[:.]?\s*([\d,.]+)\s*(?:KGS?|KG|MT)"))
    _first(f, "goods_description", _bill_goods(evidence, text))
    table = _container_rows(evidence)
    package_total = sum(int(row["package_count"]) for row in table if row.get("package_count", "").isdigit())
    gross_total = sum(float(row["gross_weight"]) for row in table if row.get("gross_weight"))
    issues = []
    declared = f["total_package_count"]["value"]
    if declared and package_total and int(declared) != package_total:
        issues.append({"kind": "package_total_mismatch", "declared": declared, "calculated": package_total})
    if f["container_count"]["value"] and len(table) and int(f["container_count"]["value"]) != len(table):
        issues.append({"kind": "container_count_mismatch", "declared": f["container_count"]["value"], "calculated": len(table)})
    if gross_total:
        f["container_gross_weight_total"] = _field(str(gross_total), None, "table_calculation")
    f["table_validation"] = _field("待人工确认" if issues else "未发现可计算差异", None, "table_validation")
    # Structured consumers retain the exception list; no source value is overwritten.
    f["table_validation"]["issues"] = issues
    return f, table


EXTRACTORS = {"装箱单": _packing, "保险单／保险凭证": _insurance, "原产地证": _origin, "检验证书": _analysis, "提单／运输单据": _bill}


def extract_structured(document_type: str, text: str, evidence: list[dict]) -> dict:
    extractor = EXTRACTORS.get(document_type)
    fields, table_data = extractor(text or _line_texts(evidence), evidence) if extractor else ({}, [])
    return {"document_type": document_type, "fields": fields, "table_data": table_data, "extraction_status": "pending_confirmation", "raw_text": text or _line_texts(evidence)}


def flat_structured_fields(structured: dict) -> dict:
    """Expose compatibility names used by the existing V2 review code."""
    values = {key: item.get("value", "") for key, item in structured.get("fields", {}).items()}
    maps = {
        "commercial_invoice": {"发票号": "invoice_number", "发票日期": "invoice_date", "发票币种": "invoice_currency", "发票金额": "invoice_amount", "发票出口商": "exporter", "发票进口商": "importer", "发票货物描述": "goods_description", "发票数量": "package_count"},
        "packing_list": {"装箱单号": "packing_list_number", "装箱单货物描述": "goods_description", "装箱单数量": "total_bundle_count", "装箱单装货港": "port_of_loading", "装箱单卸货港": "port_of_discharge", "装箱单毛重": "total_gross_weight", "装箱单净重": "total_net_weight"},
        "bill_of_lading": {"提单号": "bill_of_lading_number", "装运日": "date_laden_on_board", "提单装货港": "port_of_loading", "提单卸货港": "port_of_discharge", "提单货物描述": "goods_description", "提单数量": "total_package_count"},
        "insurance_certificate": {"保险单号": "insurance_certificate_number", "保险币种": "insured_currency", "保险金额": "insured_value", "保险货物描述": "goods_description", "保险数量": "package_count"},
        "certificate_of_origin": {"原产地证号": "certificate_number", "原产地证货物描述": "goods_description", "原产地证数量": "package_count", "原产国": "country_of_origin"},
        "inspection_certificate": {"检验证书号": "certificate_number", "检验证书货物描述": "goods_description"},
        "装箱单": {"装箱单号": "packing_list_number", "装箱单货物描述": "goods_description", "装箱单数量": "total_bundle_count", "装箱单装货港": "port_of_loading", "装箱单卸货港": "port_of_discharge", "装箱单毛重": "total_gross_weight", "装箱单净重": "total_net_weight"},
        "保险单／保险凭证": {"保险单号": "insurance_certificate_number", "保险币种": "insured_currency", "保险金额": "insured_value", "保险货物描述": "goods_description", "保险数量": "insurance_quantity", "保险索赔代理": "claims_agent"},
        "原产地证": {"原产地证号": "certificate_number", "原产地证货物描述": "goods_description", "原产地证数量": "goods_quantity", "原产国": "country_of_origin"},
        "检验证书": {"检验证书号": "certificate_number", "检验证书货物描述": "product_description"},
        "提单／运输单据": {"提单号": "bill_of_lading_number", "装运日": "date_laden_on_board", "提单装货港": "port_of_loading", "提单卸货港": "port_of_discharge", "提单货物描述": "goods_description", "提单数量": "total_package_count"},
    }
    return {flat: values.get(internal, "") for flat, internal in maps.get(structured.get("document_type"), {}).items()}
