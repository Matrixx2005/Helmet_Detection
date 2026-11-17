"""Các khung trình bày quy trình và bằng chứng cho demo một ảnh."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from helmet_ppe.config import PipelineConfig, Status
from helmet_ppe.rule_engine import RULE_DESCRIPTIONS
from helmet_ppe.schemas import ImageResult, PersonResult
from helmet_ppe.visualize import draw_analysis_stage, draw_result, person_view_bounds

STEPS = ["1 · Nhận ảnh", "2 · Nhận diện YOLO", "3 · Ghép với người", "4 · Suy luận hệ luật"]
STATUS_LABELS = {
    Status.HELMET_DETECTED: "Đội mũ",
    Status.SUSPECTED_NO_HELMET: "Nghi không đội mũ",
    Status.UNKNOWN: "Chưa đủ bằng chứng",
    Status.NO_PERSON_DETECTED: "Không có người",
}
STAGE_NOTES = [
    "Ảnh đầu vào được giữ nguyên để đối chiếu với các bước xử lý.",
    "Các hộp D… là đầu ra mô hình trước bước lọc của pipeline. "
    "Người từ mô hình pose có nhãn pose_person khi bật hỗ trợ ảnh đông người.",
    "Sau lọc và hợp nhất người, các ID P… được đánh từ trái sang phải. "
    "Đường nối biểu diễn bằng chứng được ghép. Hộp xám là vật thể chưa ghép được.",
    "Hệ luật xét bằng chứng của từng người theo thứ tự R1 → R5 và dừng tại luật đầu tiên khớp.",
]


def show_status_metrics(total: int, counts: dict[str, int], *, key: str) -> None:
    """Thống kê ưu tiên người cần kiểm tra, dùng cho một ảnh và hàng loạt."""
    with st.container(key=key):
        columns = st.columns(4)
        columns[0].metric("Nghi không đội mũ", counts[str(Status.SUSPECTED_NO_HELMET)])
        columns[1].metric("Số người", total)
        columns[2].metric("Đội mũ", counts[str(Status.HELMET_DETECTED)])
        columns[3].metric("Chưa đủ bằng chứng", counts[str(Status.UNKNOWN)])


def show_result_overview(image, result: ImageResult, cfg: PipelineConfig, result_key: str, detail) -> None:
    """Hiển thị kết luận ngay trên ảnh và đối chiếu từng người ở cột bên cạnh."""
    suspected = result.count_status(Status.SUSPECTED_NO_HELMET)
    unknown = result.count_status(Status.UNKNOWN)
    if suspected:
        st.error(f"Cần kiểm tra {suspected} người nghi không đội mũ.", icon=":material/warning:")
    elif unknown:
        st.warning(f"Có {unknown} người chưa đủ bằng chứng để kết luận.", icon=":material/help:")
    elif result.persons:
        st.success(f"Phát hiện mũ bảo hộ trên cả {len(result.persons)} người.", icon=":material/check_circle:")
    else:
        st.info("Không phát hiện người trong ảnh.", icon=":material/person_search:")

    selected_id = None
    with detail:
        if result.persons:
            with st.container(border=True, key="overview_details"):
                st.subheader("Kiểm tra từng người")
                # Ưu tiên hiển thị người nghi vi phạm trong danh sách đối chiếu.
                priority = {Status.SUSPECTED_NO_HELMET: 0, Status.UNKNOWN: 1, Status.HELMET_DETECTED: 2}
                people = sorted(result.persons, key=lambda p: (priority[p.status], p.person_id))
                by_id = {p.person_id: p for p in people}
                selected_id = st.selectbox(
                    "Người trên ảnh", [None] + list(by_id),
                    format_func=lambda value: "Toàn bộ ảnh" if value is None else
                    f"P{value} · {STATUS_LABELS[by_id[value].status]}",
                    key=f"overview_person_{result_key}")
                if selected_id is not None:
                    person = by_id[selected_id]
                    x1, y1, x2, y2 = person_view_bounds(result, selected_id)
                    if x2 > x1 and y2 > y1:
                        crop = draw_result(image, result)[y1:y2, x1:x2]
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width="stretch")
                    st.write(f"**{STATUS_LABELS[person.status]} · {person.rule_id}**")
                    st.write(person.reason)
                    with st.expander("Bằng chứng vùng đầu"):
                        show_evidence(person)
                else:
                    st.caption("Chọn một người để phóng to và kiểm tra bằng chứng.")

    canvas = draw_result(image, result) if selected_id is None else draw_analysis_stage(
        image, result, 3, cfg.association, selected_id)
    with st.container(key="result_image"):
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=result.image_name, width="stretch")
    st.caption("✓ Xanh lá · Đội mũ　⚠ Đỏ · Nghi không đội mũ　? Cam · Chưa đủ bằng chứng")


def move_step(offset: int) -> None:
    st.session_state.demo_step = max(0, min(3, st.session_state.get("demo_step", 0) + offset))


def show_evidence(person: PersonResult) -> None:
    rows = []
    for label, item in (("Mũ", person.helmet), ("Đầu không mũ", person.no_helmet)):
        rows.append({
            "Bằng chứng": label,
            "Vật thể": f"D{item.det_id}" if item else "Chưa ghép",
            "Confidence": f"{item.confidence:.4f}" if item else "—",
            "Điểm ghép": f"{item.match_score:.4f}" if item else "—",
            "Độ mạnh": f"{item.strength:.4f}" if item else "0.0000",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Độ mạnh bằng chứng = confidence phát hiện × điểm ghép hình học. "
               "Các điểm nằm trong khoảng 0–1.")


def show_rule_trace(person: PersonResult, cfg: PipelineConfig) -> None:
    h = person.helmet.strength if person.helmet else 0.0
    n = person.no_helmet.strength if person.no_helmet else 0.0
    rules = cfg.rules
    conditions = [
        ("R1", f"H ≥ {rules.min_helmet_evidence:g} và H ≥ N + {rules.decision_margin:g}",
         h >= rules.min_helmet_evidence and h >= n + rules.decision_margin),
        ("R2", f"N ≥ {rules.min_no_helmet_evidence:g} và N ≥ H + {rules.decision_margin:g}",
         n >= rules.min_no_helmet_evidence and n >= h + rules.decision_margin),
        ("R3", "H > 0 và N > 0", h > 0 and n > 0),
        ("R4", "H > 0 hoặc N > 0", h > 0 or n > 0),
        ("R5", "H = 0 và N = 0", h == 0 and n == 0),
    ]
    st.markdown(f"**H = {h:.4f}** · bằng chứng mũ &nbsp;&nbsp; **N = {n:.4f}** · bằng chứng đầu không mũ")
    rows = []
    stopped = False
    for rule, expression, matches in conditions:
        outcome = "Chưa xét" if stopped else "Không khớp"
        if rule == person.rule_id:
            outcome = "✓ Áp dụng · dừng"
        elif matches and not stopped:
            outcome = "Khớp"
        rows.append({"Luật": rule, "Điều kiện": expression, "Xét theo thứ tự": outcome})
        stopped = stopped or rule == person.rule_id
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Điều kiện dùng giá trị đầy đủ khi tính toán. Số hiển thị được làm tròn 4 chữ số.")


def show_person(person: PersonResult, stage: int, cfg: PipelineConfig) -> None:
    st.markdown(f"### Người P{person.person_id}")
    st.caption(f"Detection D{person.person_det_id} · nguồn {person.source} · "
               f"confidence người {person.person_confidence:.4f}")
    if stage == 3:
        message = f"{STATUS_LABELS[person.status]} · {person.rule_id}"
        if person.status == Status.HELMET_DETECTED:
            st.success(message)
        elif person.status == Status.SUSPECTED_NO_HELMET:
            st.error(message)
        else:
            st.warning(message)
        st.write(person.reason)
    st.markdown("**Cách ghép bằng chứng**")
    if person.head is not None:
        method = "mắt, mũi, tai" if person.head.method == "face" else "vai"
        st.write(f"Ước lượng đầu từ {method} của mô hình pose, rồi kiểm tra vị trí và kích thước vật thể.")
    else:
        st.write("Dùng vị trí tương đối, kích thước và phần nằm trong khung người để chấm điểm vùng đầu.")
    st.caption(f"Điểm ghép tối thiểu {cfg.association.min_match_score:g}. "
               "Mỗi vật thể được ghép tối đa một người trong từng lớp.")
    show_evidence(person)
    if person.flags:
        flag_labels = {"small_person": "Người có kích thước nhỏ trong ảnh",
                       "touches_top_edge": "Khung người chạm mép trên ảnh"}
        st.caption(" · ".join(flag_labels.get(flag, flag) for flag in person.flags))


def show_demo_result(image, result: ImageResult, cfg: PipelineConfig, result_key: str) -> None:
    previous, following, space = st.columns([1, 1, 4])
    previous.button("← Bước trước", on_click=move_step, args=(-1,),
                    disabled=st.session_state.get("demo_step", 0) == 0, width="stretch")
    following.button("Bước tiếp →", on_click=move_step, args=(1,),
                     disabled=st.session_state.get("demo_step", 0) == 3, width="stretch")
    space.caption("Chọn từng bước để thuyết trình trên cùng một ảnh.")
    stage = st.radio("Quy trình phân tích", range(4), format_func=lambda i: STEPS[i],
                     horizontal=True, key="demo_step", label_visibility="collapsed")
    st.info(STAGE_NOTES[stage])

    selected_id = None
    if stage >= 2 and result.persons:
        selected_id = st.selectbox(
            "Đối chiếu từng người trên ảnh", [None] + [p.person_id for p in result.persons],
            format_func=lambda value: "Toàn bộ ảnh" if value is None else f"Người P{value}",
            key=f"demo_person_{result_key}")
    canvas = draw_analysis_stage(image, result, stage, cfg.association, selected_id)
    view, detail = st.columns([1.7, 1], gap="large")
    with view:
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), width="stretch")
        if stage == 1:
            st.caption("Xanh dương · người | Xanh ngọc · mũ | Tím · đầu không mũ")
        elif stage == 2:
            st.caption("Chấm tròn · đầu từ pose | Dấu + · vị trí đầu theo khung | Đường nối · cặp đã ghép")
        elif stage == 3:
            st.caption("Xanh lá · đội mũ | Đỏ · nghi không đội mũ | Cam · chưa đủ bằng chứng")
        if selected_id is not None:
            x1, y1, x2, y2 = person_view_bounds(result, selected_id)
            with st.expander(f"Phóng to người P{selected_id}", expanded=True):
                st.image(cv2.cvtColor(canvas[y1:y2, x1:x2], cv2.COLOR_BGR2RGB),
                         width=min(480, max(240, (x2 - x1) * 2)))
    with detail:
        if stage == 0:
            st.markdown("### Ảnh đầu vào")
            st.write(result.image_name)
            st.write(f"Kích thước {result.width} × {result.height} px")
            st.write(f"Mô hình {Path(cfg.detector.model_path).name}")
            st.caption(f"Thời gian mô hình {result.inference_ms:.0f} ms · "
                       f"toàn pipeline {result.total_ms:.0f} ms")
        elif stage == 1:
            st.markdown("### Đầu ra mô hình")
            raw = result.raw_detections or result.detections
            st.write(f"{len(raw)} hộp trước lọc → {len(result.detections)} hộp sau lọc.")
            st.write("Lọc theo confidence từng lớp, loại nhiễu đầu trần dưới mũ và hợp nhất người từ PPE/pose theo cấu hình.")
            st.dataframe(pd.DataFrame([
                {"Lớp": label, "Confidence tối thiểu": value}
                for label, value in cfg.detector.class_thresholds.items()
                if label != "pose_person" or cfg.detector.use_pose
            ]), hide_index=True, width="stretch")
            with st.expander("Các hộp trước lọc"):
                st.dataframe(pd.DataFrame([
                    {"ID": f"D{d.det_id}", "Lớp": d.label, "Confidence": d.confidence,
                     "Nguồn": d.source} for d in raw
                ]), hide_index=True, width="stretch")
        elif selected_id is not None:
            person = next(p for p in result.persons if p.person_id == selected_id)
            show_person(person, stage, cfg)
        elif not result.persons:
            st.warning("Không phát hiện người trong ảnh.")
            if stage == 3:
                st.write(f"R0 · {RULE_DESCRIPTIONS['R0']}")
        elif stage == 2:
            st.markdown("### Ghép người và vùng đầu")
            st.metric("Người sau hợp nhất", len(result.persons))
            st.write(f"{sum(p.helmet is not None for p in result.persons)} cặp người–mũ · "
                     f"{sum(p.no_helmet is not None for p in result.persons)} cặp người–đầu không mũ")
            st.write(f"Vật thể chưa ghép gồm {len(result.unmatched_helmet_ids)} mũ và "
                     f"{len(result.unmatched_no_helmet_ids)} đầu không mũ.")
            st.caption("Chọn một người phía trên để xem cặp vật thể và điểm ghép cụ thể.")
        else:
            st.markdown("### Kết luận toàn ảnh")
            st.write(STATUS_LABELS[result.image_status])
            st.caption("Ưu tiên nghi không đội mũ khi có người thuộc trạng thái này, "
                       "sau đó đến chưa đủ bằng chứng, cuối cùng là đội mũ.")
            st.write("Chọn một người để xem bằng chứng và thứ tự áp dụng luật.")

    if stage == 3:
        st.divider()
        st.subheader("Giải thích hệ luật")
        if selected_id is not None:
            show_rule_trace(next(p for p in result.persons if p.person_id == selected_id), cfg)
        else:
            st.dataframe(pd.DataFrame([
                {"Luật": rule, "Ý nghĩa": description} for rule, description in RULE_DESCRIPTIONS.items()
            ]), hide_index=True, width="stretch")
