"""Bounded decimal arithmetic for generation tool calls; never execute source code."""

import ast
from decimal import Decimal


def calculate(expression: str) -> str:
    if not isinstance(expression, str) or not 0 < len(expression) <= 200:
        raise ValueError("Expression must contain 1–200 characters")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 40:
        raise ValueError("Expression is too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Decimal(ast.get_source_segment(expression, node).replace("_", ""))
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = visit(node.operand)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            else:
                raise ValueError("Only +, -, *, / are supported")
        else:
            raise ValueError("Only numeric constants and arithmetic are supported")
        if not value.is_finite() or abs(value) > Decimal("1e20"):
            raise ValueError("Calculation exceeds supported range")
        return value

    result = format(visit(tree.body), "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def overdue_fee(amount, monthly_rate, elapsed_days, grace_days, days_per_month=30):
    """Estimate pro-rata fees after a grace period; convention is explicitly returned."""
    values = [
        Decimal(str(v))
        for v in (amount, monthly_rate, elapsed_days, grace_days, days_per_month)
    ]
    amount, rate, elapsed, grace, month = values
    if (
        any(not v.is_finite() or v < 0 or v > Decimal("1e20") for v in values)
        or month == 0
        or rate > 1
    ):
        raise ValueError(
            "Require nonnegative finite values, positive month length and rate between 0 and 1"
        )
    days = max(Decimal(0), elapsed - grace)
    return {
        "overdue_days": str(days),
        "monthly_fee": calculate(f"{amount} * {rate}"),
        "estimated_fee": calculate(f"{amount} * {rate} * {days} / {month}"),
        "assumption": f"Pro-rata after the grace period; assumes {month} days/month. Policy must confirm this convention.",
    }


def grounded_fee_answer(query: str, contexts: list[str]):
    """Handle unambiguous advance-payment questions from explicit policy evidence.

    Unsupported phrasing or conflicting rules returns None for ordinary RAG.
    The 30-day convention is an estimate, never asserted as source policy.
    """
    import re

    money = re.search(r"tạm ứng\s+([\d.,]+)\s*(triệu|tỷ|nghìn|vnđ|đồng)", query.lower())
    elapsed = re.search(r"sau\s+(\d+)\s+ngày", query.lower())
    if (
        not money
        or not elapsed
        or "thanh toán" not in query.lower()
        or "công tác" in query.lower()
    ):
        return None
    policy = [
        c for c in contexts if "khoản tạm ứng" in c.lower() and "ĐÃ THAY THẾ" not in c
    ]
    evidence = "\n".join(policy)
    deadlines = set(
        re.findall(r"(?:trong vòng|sau)\s*\*{0,2}(\d+)\*{0,2}\s+ngày", evidence)
    )
    rates = set(re.findall(r"(\d+(?:[.,]\d+)?)\s*\*{0,2}\s*%\s*/\s*tháng", evidence))
    if len(deadlines) != 1 or len(rates) != 1:
        return None
    raw_amount, unit = money.groups()
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw_amount):
        raw_amount = raw_amount.replace(".", "").replace(",", "")
    else:
        raw_amount = raw_amount.replace(",", ".")
    amount = (
        Decimal(raw_amount)
        * {"triệu": 1000000, "tỷ": 1000000000, "nghìn": 1000, "vnđ": 1, "đồng": 1}[unit]
    )
    rate_percent = Decimal(next(iter(rates)).replace(",", "."))
    deadline = int(next(iter(deadlines)))
    result = overdue_fee(amount, rate_percent / 100, int(elapsed.group(1)), deadline)
    sources = list(
        dict.fromkeys(
            s
            for c in policy
            for s in re.findall(r"^\[Nguồn: ([^\]]+)\]", c, re.MULTILINE)
        )
    )
    if not sources:
        return None
    return (
        f"Ước tính phí quá hạn là **{result['estimated_fee']} VNĐ** nếu áp dụng pro-rata "
        f"và giả định tháng 30 ngày. Thanh toán sau {elapsed.group(1)} ngày, "
        f"thời hạn {deadline} ngày nên quá hạn {result['overdue_days']} ngày. "
        f"Công thức: {amount} × {rate_percent / 100} × {result['overdue_days']}/30. "
        f"Nguồn quy định {rate_percent}%/tháng ({result['monthly_fee']} VNĐ/tháng); "
        "quy ước pro-rata/30 ngày là giả định tính toán, chưa được quy định trong nguồn. "
        f"(Nguồn: {', '.join(sources)})"
    )
