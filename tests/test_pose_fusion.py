"""Kiểm thử ước lượng đầu từ điểm khớp, hợp nhất người PPE + pose, và ghép theo đầu.

Các cảnh mô phỏng lỗi thực tế ở ảnh đông người: mũ của người đứng sau lọt vào
khung người đứng trước, và người bị che khuất mà mô hình PPE bỏ sót.
"""

from dataclasses import replace

from factories import TEST_CFG, Scene, face_keypoints

from helmet_ppe.config import Status
from helmet_ppe.evaluation.person_eval import match_persons
from helmet_ppe.fusion import filter_detections, fuse_persons, suppress_faces_under_helmets
from helmet_ppe.head import estimate_head

NO_POSE_CFG = replace(TEST_CFG, detector=replace(TEST_CFG.detector, use_pose=False))


# ----------------------------------------------------------- ước lượng đầu
def test_estimate_head_from_face():
    head = estimate_head(face_keypoints(100, 50, 40))
    assert head.method == "face"
    assert abs(head.cx - 100) < 1 and abs(head.cy - 52) < 5
    assert 30 < head.size < 55


def test_no_head_estimate_when_face_hidden():
    """Quay lưng: chỉ thấy vai → không đoán vị trí đầu (ghép theo khung người)."""
    kps = [(0, 0, 0)] * 5 + [(80, 200, 0.9), (120, 200, 0.9)] + [(0, 0, 0)] * 10
    assert estimate_head(tuple(kps)) is None


def test_estimate_head_none_without_reliable_points():
    assert estimate_head(None) is None
    assert estimate_head(face_keypoints(100, 50, 40, conf=0.2)) is None


# ----------------------------------------------------------- hợp nhất người
def test_fuse_merges_overlapping_people_and_keeps_pose_only_person():
    scene = Scene()
    ppe = scene.person(100, 100, 300, 700)
    scene.pose_person(105, 95, 295, 705, head=(200, 160, 60))
    pose_only = scene.pose_person(600, 120, 760, 650, head=(680, 170, 50))
    persons, merged = fuse_persons(filter_detections(scene.detections, TEST_CFG), 0.5)
    by_id = {p.det_id: p for p in persons}
    assert len(persons) == 2
    assert by_id[ppe].source == "ppe+pose" and by_id[ppe].keypoints is not None
    assert by_id[pose_only].source == "pose" and by_id[pose_only].label == "person"
    assert ppe in merged


def test_pose_people_are_dropped_when_pose_disabled():
    scene = Scene()
    scene.pose_person(600, 120, 760, 650, head=(680, 170, 50))
    assert filter_detections(scene.detections, NO_POSE_CFG) == []


# ----------------------------------------------------------- ghép theo đầu
def test_helmet_of_person_behind_is_not_given_to_bare_headed_person_in_front():
    """Tình huống anh cảnh sát: khung người phía trước chứa mũ của người phía sau."""
    scene = Scene()
    box = (400, 100, 600, 700)
    scene.person(*box)
    scene.pose_person(*box, head=(500, 190, 60))        # mặt người phía trước
    scene.add("helmet", (540, 80, 600, 120))             # mũ người phía sau, lệch lên-phải
    with_pose = scene.analyze()
    assert with_pose.persons[0].status == Status.UNKNOWN
    assert len(with_pose.unmatched_helmet_ids) == 1

    # Không có pose thì cách ghép theo khung sẽ (sai) gán mũ này cho người phía trước.
    without_pose = Scene()
    without_pose.detections = [d for d in scene.detections if d.label != "pose_person"]
    assert without_pose.analyze().persons[0].status == Status.HELMET_DETECTED


def test_bare_head_beats_helmet_held_above_or_behind():
    """Tình huống ảnh đám đông: người không đội mũ, có mũ của người khác phía sau đầu."""
    scene = Scene()
    box = (300, 150, 560, 700)
    scene.person(*box)
    scene.pose_person(*box, head=(430, 260, 70))
    scene.add("no_helmet", (390, 200, 470, 300), conf=0.6)   # đầu trần đúng chỗ
    scene.add("helmet", (300, 60, 380, 120), conf=0.8)        # mũ của người phía sau
    person = scene.analyze().persons[0]
    assert person.status == Status.SUSPECTED_NO_HELMET
    assert person.helmet is None


def test_helmet_on_head_is_matched_by_head_method():
    scene = Scene()
    scene.pose_person(600, 120, 760, 650, head=(680, 190, 50))
    scene.add("helmet", (650, 130, 710, 175))
    person = scene.analyze().persons[0]
    assert person.status == Status.HELMET_DETECTED
    assert person.source == "pose" and person.head is not None


def test_eval_matches_people_by_head_box_first():
    pred = {"image_name": "i", "person_id": 1, "x1": 0, "y1": 0, "x2": 300, "y2": 900,
            "head_x1": 100, "head_y1": 10, "head_x2": 160, "head_y2": 60, "status": "HELMET_DETECTED"}
    gt = {"image_name": "i", "gt_person_id": 1, "x1": 80, "y1": 0, "x2": 200, "y2": 300,
          "head_x1": 105, "head_y1": 12, "head_x2": 158, "head_y2": 58, "gt_status": "HELMET"}
    matches, missed, extra = match_persons([pred], [gt], 0.5)  # IoU khung người < 0.5
    assert len(matches) == 1 and not missed and not extra


# ----------------------------------------------------------- mặt dưới mũ
def test_face_box_under_helmet_is_not_a_violation():
    """Ảnh đông người: mô hình bắn no_helmet (conf thấp) vào mặt ngay dưới mũ."""
    scene = Scene()
    scene.pose_person(480, 220, 580, 480, head=(530, 280, 20))
    scene.add("helmet", (502, 222, 552, 253), conf=0.37)
    scene.add("no_helmet", (486, 257, 574, 307), conf=0.08)
    result = scene.analyze()
    assert result.persons[0].status != Status.SUSPECTED_NO_HELMET
    assert all(d.label != "no_helmet" for d in result.detections)

    off = replace(TEST_CFG, association=replace(TEST_CFG.association,
                                                suppress_faces_under_helmets=False))
    assert any(d.label == "no_helmet" for d in scene.analyze(off).detections)


def test_bare_head_beside_helmet_is_kept():
    scene = Scene()
    scene.add("helmet", (100, 100, 160, 140), conf=0.9)
    scene.add("no_helmet", (220, 110, 280, 170), conf=0.3)   # người bên cạnh
    kept = suppress_faces_under_helmets(scene.detections)
    assert any(d.label == "no_helmet" for d in kept)
