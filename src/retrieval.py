"""Cover query facets and select policy versions by effective date."""

import re
from datetime import date, datetime
from zoneinfo import ZoneInfo


def policy_metadata(document):
    title = document["text"].splitlines()[0].lstrip("# ")
    family = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip().lower()
    match = re.search(r"Ngày hiệu lực:\s*(\d{2}/\d{2}/\d{4})", document["text"])
    effective = (
        date(*map(int, reversed(match.group(1).split("/")))).isoformat()
        if match
        else None
    )
    return {
        **document["metadata"],
        "policy_family": family,
        "effective_date": effective,
    }


def query_facets(query):
    parts = re.split(r"\s+và\s+", query, maxsplit=1, flags=re.IGNORECASE)
    if (
        len(parts) == 2
        and len(parts[1].split()) >= 3
        and re.search(r"\b(?:nào|bao nhiêu|ai|gì|cần)\b", parts[1], re.IGNORECASE)
    ):
        return [part.strip() for part in parts]
    return []


def valid_policy(result, query, versions):
    family, effective = (
        result.metadata.get("policy_family"),
        result.metadata.get("effective_date"),
    )
    if not family or not effective or family not in versions:
        return True
    year = re.search(r"\b(20\d{2})\b", query)
    cutoff = (
        f"{year.group(1)}-12-31"
        if year
        else datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat()
    )
    applicable = [d for d in versions[family] if d and d <= cutoff]
    return bool(applicable) and effective == max(applicable)
