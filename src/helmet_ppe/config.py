"""Hằng số và cấu hình dùng chung cho toàn bộ hệ thống."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# Tên lớp (chữ thường) mà pipeline sử dụng. Mô hình Construction-PPE có 11 lớp,
# nhưng phạm vi đề tài chỉ dùng người, mũ và "không đội mũ".
PERSON = "person"
HELMET = "helmet"
NO_HELMET = "no_helmet"
RELEVANT_CLASSES = (PERSON, HELMET, NO_HELMET)


class Status(str, Enum):
    """Trạng thái đội mũ của một người (hoặc của ảnh khi không có người)."""

    HELMET_DETECTED = "HELMET_DETECTED"
    SUSPECTED_NO_HELMET = "SUSPECTED_NO_HELMET"
    UNKNOWN = "UNKNOWN"
    NO_PERSON_DETECTED = "NO_PERSON_DETECTED"

    def __str__(self) -> str:  # để ghi CSV/JSON gọn gàng
        return self.value


PERSON_STATUSES = (Status.HELMET_DETECTED, Status.SUSPECTED_NO_HELMET, Status.UNKNOWN)


@dataclass(frozen=True)
class DetectorConfig:
    """Tham số suy luận YOLO."""

    model_path: str = str(DEFAULT_MODEL_PATH)
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str | None = None  # None = để Ultralytics tự chọn (GPU nếu có)
    only_relevant_classes: bool = True


@dataclass(frozen=True)
class AssociationConfig:
    """Tham số hình học để ghép mũ/không-mũ với người.

    Tọa độ tương đối được tính theo khung người: ``rel_y = (cy - y1) / h`` và
    ``rel_x = (cx - x_tâm) / (w / 2)``. Các giá trị mặc định được hiệu chỉnh từ
    nhãn chuẩn Construction-PPE (train + val): tâm mũ nằm ở trung vị
    ``rel_y ≈ 0.11``, 95% mẫu có ``rel_y ≤ 0.27``.
    """

    expected_rel_y: float = 0.11   # vị trí đầu kỳ vọng theo chiều dọc
    vertical_tolerance: float = 0.25  # độ lệch dọc để điểm dọc giảm về 0
    min_rel_y: float = -0.15       # mũ được phép nhô lên trên khung người
    max_rel_y: float = 0.40        # thấp hơn mức này: không còn là vùng đầu (vd. cầm trên tay)
    max_abs_rel_x: float = 1.05    # tâm mũ phải nằm trong bề ngang khung người
    min_width_ratio: float = 0.06  # mũ quá nhỏ so với người: nghi nhiễu
    max_width_ratio: float = 1.30  # mũ quá to so với người: không hợp lý
    min_inside_ratio: float = 0.50  # tỉ lệ diện tích mũ nằm trong khung người (đã nới lên trên)
    min_match_score: float = 0.35  # dưới ngưỡng này không ghép
    weight_vertical: float = 0.5
    weight_horizontal: float = 0.3
    weight_inside: float = 0.2


@dataclass(frozen=True)
class RuleConfig:
    """Ngưỡng của hệ luật. Bằng chứng = confidence × điểm ghép."""

    min_helmet_evidence: float = 0.25
    min_no_helmet_evidence: float = 0.25
    decision_margin: float = 0.10  # chênh lệch tối thiểu khi có cả hai loại bằng chứng
    min_person_height_px: float = 40.0  # người quá nhỏ → khó kết luận
    edge_margin_px: float = 2.0  # khung người chạm mép trên → có thể mất phần đầu


@dataclass(frozen=True)
class PipelineConfig:
    """Gộp toàn bộ cấu hình của một lần chạy (được ghi vào config.json)."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)

    def to_dict(self) -> dict:
        return asdict(self)
