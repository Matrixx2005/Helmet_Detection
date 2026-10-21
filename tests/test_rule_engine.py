"""Kiểm thử hệ luật: các tình huống bắt buộc của đề bài và từng luật R0–R5."""

import pytest
from factories import Scene

from helmet_ppe.config import RuleConfig, Status
from helmet_ppe.rule_engine import FLAG_SMALL, FLAG_TOP_EDGE, decide

CFG = RuleConfig()


def statuses(result):
    return [p.status for p in result.persons]


# ----------------------------------------------------------- tình huống của đề bài
def test_one_person_wearing_helmet():
    scene = Scene()
    box = (400, 100, 600, 700)
    scene.person(*box)
    scene.helmet_on(box)
    result = scene.analyze()
    assert statuses(result) == [Status.HELMET_DETECTED]
    assert result.persons[0].rule_id == "R1"
    assert result.image_status == Status.HELMET_DETECTED


def test_one_person_without_visible_helmet_is_unknown_not_violation():
    scene = Scene()
    scene.person(400, 100, 600, 700)
    result = scene.analyze()
    assert statuses(result) == [Status.UNKNOWN]
    assert result.persons[0].rule_id == "R5"


def test_two_people_side_by_side_both_with_helmets():
    scene = Scene()
    for box in [(100, 100, 300, 700), (320, 100, 520, 700)]:
        scene.person(*box)
        scene.helmet_on(box)
    assert statuses(scene.analyze()) == [Status.HELMET_DETECTED] * 2


def test_one_with_helmet_one_without():
    scene = Scene()
    left, right = (100, 100, 300, 700), (500, 100, 700, 700)
    scene.person(*left)
    scene.helmet_on(left)
    scene.person(*right)
    scene.no_helmet_on(right)
    result = scene.analyze()
    assert statuses(result) == [Status.HELMET_DETECTED, Status.SUSPECTED_NO_HELMET]
    assert result.image_status == Status.SUSPECTED_NO_HELMET


def test_person_holding_helmet_in_hand_is_not_helmet_detected():
    scene = Scene()
    scene.person(400, 100, 600, 700)
    scene.add("helmet", (560, 400, 640, 470))
    result = scene.analyze()
    assert statuses(result) == [Status.UNKNOWN]
    assert len(result.unmatched_helmet_ids) == 1


def test_helmet_not_belonging_to_anyone():
    scene = Scene()
    scene.add("helmet", (800, 650, 880, 720))
    result = scene.analyze()
    assert result.persons == []
    assert result.image_status == Status.NO_PERSON_DETECTED
    assert len(result.unmatched_helmet_ids) == 1


def test_occluded_person_cut_at_top_edge_is_unknown_with_flag():
    scene = Scene()
    scene.person(400, 0, 600, 500)  # đầu nằm ngoài khung hình
    person = scene.analyze().persons[0]
    assert person.status == Status.UNKNOWN
    assert FLAG_TOP_EDGE in person.flags
    assert "mép trên" in person.reason


def test_tiny_person_is_flagged():
    scene = Scene()
    scene.person(400, 300, 415, 330)
    assert FLAG_SMALL in scene.analyze().persons[0].flags


def test_image_without_person():
    result = Scene().analyze()
    assert result.persons == []
    assert result.image_status == Status.NO_PERSON_DETECTED


# ----------------------------------------------------------- từng luật
@pytest.mark.parametrize(
    ("helmet", "no_helmet", "expected"),
    [
        (0.80, 0.00, (Status.HELMET_DETECTED, "R1")),
        (0.80, 0.50, (Status.HELMET_DETECTED, "R1")),
        (0.00, 0.80, (Status.SUSPECTED_NO_HELMET, "R2")),
        (0.30, 0.75, (Status.SUSPECTED_NO_HELMET, "R2")),
        (0.60, 0.55, (Status.UNKNOWN, "R3")),
        (0.10, 0.00, (Status.UNKNOWN, "R4")),
        (0.00, 0.15, (Status.UNKNOWN, "R4")),
        (0.00, 0.00, (Status.UNKNOWN, "R5")),
    ],
)
def test_decide_rule_table(helmet, no_helmet, expected):
    assert decide(helmet, no_helmet, CFG) == expected


def test_low_confidence_helmet_gives_weak_evidence():
    scene = Scene()
    box = (400, 100, 600, 700)
    scene.person(*box)
    scene.helmet_on(box, conf=0.2)
    person = scene.analyze().persons[0]
    assert (person.status, person.rule_id) == (Status.UNKNOWN, "R4")


def test_conflicting_helmet_and_no_helmet_is_unknown():
    scene = Scene()
    box = (400, 100, 600, 700)
    scene.person(*box)
    scene.helmet_on(box, conf=0.6)
    scene.no_helmet_on(box, conf=0.6)
    person = scene.analyze().persons[0]
    assert (person.status, person.rule_id) == (Status.UNKNOWN, "R3")


def test_person_ids_start_at_one_and_are_unique():
    scene = Scene()
    for x in (100, 400, 700):
        scene.person(x, 100, x + 150, 600)
    assert [p.person_id for p in scene.analyze().persons] == [1, 2, 3]
