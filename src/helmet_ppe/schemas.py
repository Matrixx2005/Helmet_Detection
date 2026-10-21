"""Cấu trúc dữ liệu truyền giữa các module.

Tất cả module dùng chung các dataclass này thay vì dict tự do, giúp dữ liệu
nhất quán và dễ xuất ra CSV/JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from helmet_ppe import geometry
from helmet_ppe.config import Status
from helmet_ppe.geometry import BBox


@dataclass(frozen=True)
class Detection:
    """Một đối tượng do YOLO phát hiện."""

    det_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox

    @property
    def label(self) -> str:
        """Tên lớp chuẩn hóa chữ thường (mô hình dùng 'Person' viết hoa)."""
        return self.class_name.lower()

    @property
    def area(self) -> float:
        return geometry.area(self.bbox)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = [round(v, 2) for v in self.bbox]
        data["confidence"] = round(self.confidence, 4)
        return data


@dataclass(frozen=True)
class Candidate:
    """Một cặp (người, vật thể vùng đầu) hợp lệ về hình học kèm điểm ghép."""

    person_det_id: int
    object_det_id: int
    score: float
    rel_x: float
    rel_y: float
    width_ratio: float
    inside_ratio: float


@dataclass
class HeadEvidence:
    """Bằng chứng ở vùng đầu đã được ghép cho một người."""

    det_id: int
    confidence: float
    match_score: float
    bbox: BBox

    @property
    def strength(self) -> float:
        """Độ mạnh bằng chứng = độ tin cậy YOLO × điểm ghép hình học."""
        return self.confidence * self.match_score


@dataclass
class Association:
    """Kết quả ghép cho một người: mũ và/hoặc vùng đầu không mũ."""

    person: Detection
    helmet: HeadEvidence | None = None
    no_helmet: HeadEvidence | None = None


@dataclass
class AssociationResult:
    """Đầu ra của bước ghép cho cả ảnh."""

    associations: list[Association]
    unmatched_helmet_ids: list[int] = field(default_factory=list)
    unmatched_no_helmet_ids: list[int] = field(default_factory=list)


@dataclass
class PersonResult:
    """Kết luận cuối cùng cho một người."""

    person_id: int
    person_det_id: int
    bbox: BBox
    person_confidence: float
    status: Status
    rule_id: str
    reason: str
    helmet: HeadEvidence | None = None
    no_helmet: HeadEvidence | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def evidence(item: HeadEvidence | None) -> dict | None:
            if item is None:
                return None
            return {
                "det_id": item.det_id,
                "confidence": round(item.confidence, 4),
                "match_score": round(item.match_score, 4),
                "strength": round(item.strength, 4),
                "bbox": [round(v, 2) for v in item.bbox],
            }

        return {
            "person_id": self.person_id,
            "person_det_id": self.person_det_id,
            "bbox": [round(v, 2) for v in self.bbox],
            "person_confidence": round(self.person_confidence, 4),
            "status": str(self.status),
            "rule_id": self.rule_id,
            "reason": self.reason,
            "helmet": evidence(self.helmet),
            "no_helmet": evidence(self.no_helmet),
            "flags": list(self.flags),
        }


@dataclass
class ImageResult:
    """Toàn bộ kết quả của một ảnh."""

    image_name: str
    image_path: str
    width: int
    height: int
    detections: list[Detection]
    persons: list[PersonResult]
    image_status: Status | str
    unmatched_helmet_ids: list[int] = field(default_factory=list)
    unmatched_no_helmet_ids: list[int] = field(default_factory=list)
    inference_ms: float = 0.0
    total_ms: float = 0.0

    def count(self, label: str) -> int:
        return sum(1 for det in self.detections if det.label == label)

    def count_status(self, status: Status) -> int:
        return sum(1 for person in self.persons if person.status == status)

    def to_dict(self) -> dict:
        return {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "image_status": str(self.image_status),
            "inference_ms": round(self.inference_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "detections": [det.to_dict() for det in self.detections],
            "persons": [person.to_dict() for person in self.persons],
            "unmatched_helmet_ids": list(self.unmatched_helmet_ids),
            "unmatched_no_helmet_ids": list(self.unmatched_no_helmet_ids),
        }
