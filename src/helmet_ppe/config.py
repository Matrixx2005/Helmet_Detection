"""Hằng số và cấu hình dùng chung cho toàn bộ hệ thống."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_POSE_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n-pose.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# Tên lớp (chữ thường) mà pipeline sử dụng. Mô hình Construction-PPE có 11 lớp,
# nhưng phạm vi đề tài chỉ dùng người, mũ và "không đội mũ".
PERSON = "person"
HELMET = "helmet"
NO_HELMET = "no_helmet"
RELEVANT_CLASSES = (PERSON, HELMET, NO_HELMET)
POSE_PERSON = "pose_person"  # người do mô hình pose (COCO) phát hiện, kèm điểm khớp


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
    """Tham số suy luận YOLO.

    Mỗi lớp có ngưỡng confidence riêng. Lớp ``no_helmet`` hiếm trong dữ liệu
    huấn luyện nên mô hình cho confidence thấp (recall = 0 ở ngưỡng 0.25); ngưỡng
    riêng thấp hơn được chọn trên tập validation (xem docs/05_experiments.md).
    Hộp ``no_helmet`` độ tin cậy thấp chỉ được dùng khi ghép được vào vùng đầu
    của một người, nên bước association đóng vai trò bộ lọc nhiễu.
    """

    model_path: str = str(DEFAULT_MODEL_PATH)
    conf_person: float = 0.50
    conf_helmet: float = 0.30
    conf_no_helmet: float = 0.05
    iou: float = 0.7
    imgsz: int = 640
    device: str | None = None  # None = để Ultralytics tự chọn (GPU nếu có)
    # Mô hình pose (YOLO26n-pose, huấn luyện sẵn trên COCO) bổ sung người trong đám
    # đông và cho vị trí đầu thật (mắt, mũi, tai) để ghép mũ chính xác hơn.
    use_pose: bool = True
    pose_model_path: str = str(DEFAULT_POSE_MODEL_PATH)
    conf_pose_person: float = 0.30

    @property
    def class_thresholds(self) -> dict[str, float]:
        return {PERSON: self.conf_person, HELMET: self.conf_helmet,
                NO_HELMET: self.conf_no_helmet, POSE_PERSON: self.conf_pose_person}

    @property
    def min_conf(self) -> float:
        """Ngưỡng truyền cho YOLO; lọc theo từng lớp được làm sau đó."""
        return min(self.class_thresholds.values())


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

    # --- Ghép theo đầu ước lượng từ điểm khớp (khi người có keypoints) ---
    # Đơn vị là "kích thước đầu" s (≈ bề rộng đầu). Tâm mũ kỳ vọng nằm trên tâm
    # mặt khoảng 0,6·s; tâm hộp no_helmet (đầu trần, gồm cả tóc) trên khoảng 0,3·s.
    # Vùng chấp nhận là hình elip: rộng theo chiều ngang (mặt quay nghiêng làm tâm
    # mặt lệch sang bên) nhưng hẹp theo chiều dọc (mũ không thể "lơ lửng" trên đầu —
    # mũ cao hơn nhiều thường là của người đứng phía sau).
    keypoint_conf: float = 0.5        # điểm khớp có độ tin cậy thấp hơn bị bỏ qua
    helmet_head_offset: float = 0.6
    no_helmet_head_offset: float = 0.3
    head_tolerance_x: float = 0.9     # bán trục ngang của elip (đơn vị s)
    head_tolerance_y: float = 0.8     # bán trục dọc của elip (đơn vị s)
    min_head_size_ratio: float = 0.8  # mũ/đầu nhỏ hơn 0,8·s: không phải của đầu này
    # Điểm khớp trên mặt nhỏ nên s hay bị ước lượng thiếu: dùng s' = max(s, bề rộng vật thể / k)
    # với k = tỉ lệ bề rộng mũ / đầu (mũ có vành) và bề rộng hộp đầu trần / đầu.
    helmet_to_head_ratio: float = 1.8
    no_helmet_to_head_ratio: float = 1.3
    person_merge_iou: float = 0.5     # IoU để coi người PPE và người pose là một
    # Bỏ hộp no_helmet nằm ngay dưới một hộp helmet tin cậy hơn (khuôn mặt dưới mũ)
    suppress_faces_under_helmets: bool = True


@dataclass(frozen=True)
class RuleConfig:
    """Ngưỡng của hệ luật. Bằng chứng = confidence × điểm ghép.

    Giá trị mặc định được chọn bằng dò lưới trên tập validation.
    """

    min_helmet_evidence: float = 0.10
    min_no_helmet_evidence: float = 0.05
    decision_margin: float = 0.0  # chênh lệch tối thiểu khi có cả hai loại bằng chứng
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
