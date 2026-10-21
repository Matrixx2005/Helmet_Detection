"""Hệ thống phát hiện mũ bảo hộ và suy luận vi phạm bằng hệ luật.

Luồng xử lý: Ảnh → YOLO (detector) → ghép người–mũ (association)
→ hệ luật (rule_engine) → kết quả (ImageResult).
"""

from helmet_ppe.config import AssociationConfig, DetectorConfig, RuleConfig, Status
from helmet_ppe.schemas import Detection, ImageResult, PersonResult

__all__ = [
    "AssociationConfig",
    "Detection",
    "DetectorConfig",
    "ImageResult",
    "PersonResult",
    "RuleConfig",
    "Status",
]

__version__ = "1.0.0"
