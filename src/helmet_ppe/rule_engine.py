"""Hệ luật suy luận trạng thái đội mũ cho từng người.

Nguyên tắc thận trọng: *không thấy mũ* KHÔNG đồng nghĩa với *vi phạm*.
Chỉ kết luận ``SUSPECTED_NO_HELMET`` khi mô hình phát hiện tích cực vùng đầu
không đội mũ (lớp ``no_helmet``) và bằng chứng này vượt trội so với mũ.

Bằng chứng của một vật thể = confidence YOLO × điểm ghép hình học.

Bảng luật (áp dụng theo thứ tự, luật đầu tiên khớp sẽ quyết định):

    R0  Ảnh không có người                             → NO_PERSON_DETECTED
    R1  Mũ đủ mạnh và vượt no_helmet ít nhất một biên  → HELMET_DETECTED
    R2  no_helmet đủ mạnh và vượt mũ ít nhất một biên  → SUSPECTED_NO_HELMET
    R3  Có cả mũ và no_helmet nhưng không bên nào trội  → UNKNOWN (mâu thuẫn)
    R4  Chỉ có bằng chứng yếu (dưới ngưỡng)            → UNKNOWN (bằng chứng yếu)
    R5  Không có bằng chứng nào ở vùng đầu             → UNKNOWN (thiếu bằng chứng)
"""

from __future__ import annotations

from helmet_ppe import geometry
from helmet_ppe.config import RuleConfig, Status
from helmet_ppe.schemas import Association, PersonResult

RULE_DESCRIPTIONS: dict[str, str] = {
    "R0": "Không phát hiện người trong ảnh.",
    "R1": "Có mũ được ghép vào vùng đầu với bằng chứng đủ mạnh.",
    "R2": "Mô hình phát hiện vùng đầu không đội mũ (no_helmet) với bằng chứng đủ mạnh.",
    "R3": "Vùng đầu có cả bằng chứng mũ và không-mũ, không bên nào trội: cần xem lại.",
    "R4": "Chỉ có bằng chứng yếu ở vùng đầu, chưa đủ để kết luận.",
    "R5": "Không thấy mũ cũng không thấy no_helmet ở vùng đầu: không đủ bằng chứng.",
}

# Cờ mô tả tình huống khó, giúp giải thích các ca UNKNOWN.
FLAG_SMALL = "small_person"
FLAG_TOP_EDGE = "touches_top_edge"


def _flags(assoc: Association, cfg: RuleConfig) -> list[str]:
    flags = []
    if geometry.height(assoc.person.bbox) < cfg.min_person_height_px:
        flags.append(FLAG_SMALL)
    if assoc.person.bbox[1] <= cfg.edge_margin_px:
        flags.append(FLAG_TOP_EDGE)
    return flags


def decide(helmet_strength: float, no_helmet_strength: float,
           cfg: RuleConfig) -> tuple[Status, str]:
    """Áp dụng R1–R5 cho một người dựa trên độ mạnh hai loại bằng chứng."""
    h, n = helmet_strength, no_helmet_strength
    if h >= cfg.min_helmet_evidence and h >= n + cfg.decision_margin:
        return Status.HELMET_DETECTED, "R1"
    if n >= cfg.min_no_helmet_evidence and n >= h + cfg.decision_margin:
        return Status.SUSPECTED_NO_HELMET, "R2"
    if h > 0 and n > 0:
        return Status.UNKNOWN, "R3"
    if h > 0 or n > 0:
        return Status.UNKNOWN, "R4"
    return Status.UNKNOWN, "R5"


def _reason(rule_id: str, h: float, n: float, flags: list[str]) -> str:
    parts = [RULE_DESCRIPTIONS[rule_id]]
    if h > 0 or n > 0:
        parts.append(f"(bằng chứng mũ={h:.2f}, không-mũ={n:.2f})")
    if rule_id in {"R4", "R5"} and FLAG_SMALL in flags:
        parts.append("Người quá nhỏ trong ảnh.")
    if rule_id in {"R4", "R5"} and FLAG_TOP_EDGE in flags:
        parts.append("Khung người chạm mép trên, có thể bị cắt mất đầu.")
    return " ".join(parts)


def evaluate_person(person_id: int, assoc: Association,
                    cfg: RuleConfig | None = None) -> PersonResult:
    """Suy luận trạng thái của một người từ kết quả ghép."""
    cfg = cfg or RuleConfig()
    h = assoc.helmet.strength if assoc.helmet else 0.0
    n = assoc.no_helmet.strength if assoc.no_helmet else 0.0
    status, rule_id = decide(h, n, cfg)
    flags = _flags(assoc, cfg)
    return PersonResult(
        person_id=person_id,
        person_det_id=assoc.person.det_id,
        bbox=assoc.person.bbox,
        person_confidence=assoc.person.confidence,
        status=status,
        rule_id=rule_id,
        reason=_reason(rule_id, h, n, flags),
        helmet=assoc.helmet,
        no_helmet=assoc.no_helmet,
        flags=flags,
    )


def evaluate_image(associations: list[Association],
                   cfg: RuleConfig | None = None) -> tuple[list[PersonResult], Status]:
    """Suy luận cho mọi người trong ảnh và trạng thái tổng của ảnh.

    Trạng thái ảnh: ``NO_PERSON_DETECTED`` nếu không có người (R0); nếu có ít
    nhất một người nghi vi phạm → ``SUSPECTED_NO_HELMET``; nếu còn người chưa
    kết luận → ``UNKNOWN``; ngược lại → ``HELMET_DETECTED``.
    """
    persons = [evaluate_person(i, assoc, cfg)
               for i, assoc in enumerate(associations, start=1)]
    return persons, summarize_image(persons)


def summarize_image(persons: list[PersonResult]) -> Status:
    statuses = {p.status for p in persons}
    if not persons:
        return Status.NO_PERSON_DETECTED
    if Status.SUSPECTED_NO_HELMET in statuses:
        return Status.SUSPECTED_NO_HELMET
    if Status.UNKNOWN in statuses:
        return Status.UNKNOWN
    return Status.HELMET_DETECTED
