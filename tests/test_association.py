"""Kiểm thử thuật toán ghép người–mũ với các tình huống thực tế."""

from factories import Scene

from helmet_ppe.association import associate, score_pair
from helmet_ppe.config import AssociationConfig
from helmet_ppe.schemas import Detection

CFG = AssociationConfig()


def _helmet_of(result, person_det_id):
    return next(a.helmet for a in result.associations if a.person.det_id == person_det_id)


def test_single_person_with_helmet_is_matched():
    scene = Scene()
    box = (400, 100, 600, 700)
    pid = scene.person(*box)
    hid = scene.helmet_on(box)
    result = associate(scene.detections, CFG)
    assert _helmet_of(result, pid).det_id == hid
    assert result.unmatched_helmet_ids == []


def test_two_people_side_by_side_each_get_their_own_helmet():
    scene = Scene()
    left, right = (100, 100, 300, 700), (320, 110, 520, 710)
    p_left, p_right = scene.person(*left), scene.person(*right)
    h_right = scene.helmet_on(right)   # thêm mũ phải trước để thứ tự không giúp thuật toán
    h_left = scene.helmet_on(left)
    result = associate(scene.detections, CFG)
    assert _helmet_of(result, p_left).det_id == h_left
    assert _helmet_of(result, p_right).det_id == h_right


def test_overlapping_people_helmet_goes_to_best_aligned_head():
    scene = Scene()
    front, back = (300, 150, 520, 760), (380, 100, 600, 700)  # hai khung chồng lấn mạnh
    p_front, p_back = scene.person(*front), scene.person(*back)
    hid = scene.helmet_on(back)
    result = associate(scene.detections, CFG)
    assert _helmet_of(result, p_back).det_id == hid
    assert _helmet_of(result, p_front) is None


def test_one_helmet_is_never_assigned_to_two_people():
    scene = Scene()
    scene.person(300, 100, 500, 700)
    scene.person(320, 100, 520, 700)
    scene.add("helmet", (370, 90, 450, 200))
    result = associate(scene.detections, CFG)
    assert sum(a.helmet is not None for a in result.associations) == 1


def test_helmet_held_in_hand_is_not_matched():
    scene = Scene()
    pid = scene.person(400, 100, 600, 700)
    hid = scene.add("helmet", (560, 400, 640, 470))  # ngang thắt lưng, cạnh tay
    result = associate(scene.detections, CFG)
    assert _helmet_of(result, pid) is None
    assert result.unmatched_helmet_ids == [hid]


def test_helmet_far_from_everyone_is_unmatched():
    scene = Scene()
    scene.person(100, 100, 250, 600)
    hid = scene.add("helmet", (800, 650, 880, 720))  # mũ để dưới đất
    result = associate(scene.detections, CFG)
    assert result.unmatched_helmet_ids == [hid]


def test_no_helmet_evidence_is_matched_separately():
    scene = Scene()
    box = (400, 100, 600, 700)
    pid = scene.person(*box)
    nid = scene.no_helmet_on(box)
    assoc = associate(scene.detections, CFG).associations[0]
    assert assoc.person.det_id == pid
    assert assoc.no_helmet.det_id == nid
    assert assoc.helmet is None


def test_persons_are_ordered_left_to_right():
    scene = Scene()
    right = scene.person(600, 100, 800, 700)
    left = scene.person(100, 100, 300, 700)
    ids = [a.person.det_id for a in associate(scene.detections, CFG).associations]
    assert ids == [left, right]


def test_score_pair_rewards_centered_helmet_more_than_offset_one():
    person = Detection(0, 6, "Person", 0.9, (400, 100, 600, 700))
    centered = Detection(1, 0, "helmet", 0.9, (465, 90, 535, 200))
    offset = Detection(2, 0, "helmet", 0.9, (530, 90, 600, 200))
    assert score_pair(person, centered, CFG).score > score_pair(person, offset, CFG).score


def test_score_pair_rejects_degenerate_person_box():
    person = Detection(0, 6, "Person", 0.9, (400, 100, 400, 700))
    helmet = Detection(1, 0, "helmet", 0.9, (390, 90, 410, 120))
    assert score_pair(person, helmet, CFG) is None
