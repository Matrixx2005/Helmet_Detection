"""Ước lượng vị trí và kích thước đầu từ 17 điểm khớp COCO.

Chỉ số điểm khớp COCO dùng ở đây: 0 mũi, 1–2 mắt, 3–4 tai, 5–6 vai.

Kích thước điểm khớp trên mặt nhỏ nên ước lượng thường bị thiếu; bước ghép vì vậy
dùng thêm kích thước của chính hộp mũ để chuẩn hóa (``association._score_by_head``).

* Nếu thấy ít nhất hai điểm trên mặt: tâm đầu = trung bình các điểm mặt; kích
  thước đầu lấy giá trị lớn nhất của các ước lượng từ mặt (khoảng cách hai tai
  × 1,0; hai mắt × 2,4; mũi–tai × 1,2) — khi quay nghiêng, khoảng cách hai mắt/
  hai tai co lại nhưng mũi–tai thì không, nên lấy giá trị lớn nhất. Bề rộng vai
  chỉ dùng khi không có ước lượng nào từ mặt (vai dễ sai khi giơ tay).
* Ngược lại (quay lưng, mặt bị che): không ước lượng — bước ghép dùng cách theo
  khung người. (Thử nghiệm suy vị trí đầu từ hai vai cho sai lệch có hệ thống
  ~0,6 kích thước đầu trên tập val nên đã bỏ.)
"""

from __future__ import annotations

import math

from helmet_ppe.schemas import HeadEstimate, Keypoints

NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER = range(7)
FACE_POINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
MIN_HEAD_SIZE = 6.0  # pixel


def _visible(keypoints: Keypoints, index: int, min_conf: float) -> tuple[float, float] | None:
    if index >= len(keypoints):
        return None
    x, y, conf = keypoints[index]
    return (x, y) if conf >= min_conf else None


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def estimate_head(keypoints: Keypoints | None, min_conf: float = 0.5) -> HeadEstimate | None:
    """Trả về HeadEstimate hoặc None nếu không đủ điểm khớp tin cậy."""
    if not keypoints:
        return None
    face = [p for i in FACE_POINTS if (p := _visible(keypoints, i, min_conf)) is not None]
    shoulders = [_visible(keypoints, LEFT_SHOULDER, min_conf),
                 _visible(keypoints, RIGHT_SHOULDER, min_conf)]
    shoulder_width = _distance(*shoulders)

    if len(face) >= 2:
        cx = sum(p[0] for p in face) / len(face)
        cy = sum(p[1] for p in face) / len(face)
        estimates = []
        if ears := _distance(_visible(keypoints, LEFT_EAR, min_conf),
                             _visible(keypoints, RIGHT_EAR, min_conf)):
            estimates.append(ears)
        if eyes := _distance(_visible(keypoints, LEFT_EYE, min_conf),
                             _visible(keypoints, RIGHT_EYE, min_conf)):
            estimates.append(2.4 * eyes)
        nose = _visible(keypoints, NOSE, min_conf)
        nose_ear = [d for ear in (LEFT_EAR, RIGHT_EAR)
                    if (d := _distance(nose, _visible(keypoints, ear, min_conf)))]
        if nose_ear:
            estimates.append(1.2 * max(nose_ear))
        if not estimates and shoulder_width:
            estimates.append(0.5 * shoulder_width)
        if not estimates:  # chỉ còn hai điểm mặt không thuộc các cặp trên
            estimates.append(2.0 * max(_distance(a, b) for a in face for b in face))
        return HeadEstimate(cx, cy, max(MIN_HEAD_SIZE, *estimates), "face")

    return None
