"""Giao diện so sánh trực tiếp và đánh giá với nhãn chuẩn."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd
import streamlit as st

from helmet_ppe.batch import run_batch
from helmet_ppe.comparison import (COMPARISON_FIELDS, STATUS_LABELS, ComparisonReport,
                                   compare_results, evaluate_comparison, read_comparison_gt)
from helmet_ppe.config import PROJECT_ROOT, Status
from helmet_ppe.image_io import decode_image, list_images, read_image
from helmet_ppe.pipeline import HelmetPipeline
from helmet_ppe.visualize import draw_result, person_view_bounds

BEFORE = "Trước · Ghép đơn giản"
AFTER = "Sau · Ghép vùng đầu + hệ luật"
ERROR_LABELS = {
    "FALSE_VIOLATION": "Người đội mũ bị cảnh báo",
    "MISSED_VIOLATION": "Người không đội mũ bị kết luận đội mũ",
    "VIOLATION_AS_UNKNOWN": "Người không đội mũ chưa được kết luận",
    "HELMET_AS_UNKNOWN": "Người đội mũ chưa được kết luận",
    "MISSED_PERSON": "Người chuẩn bị bỏ sót",
    "FALSE_PERSON": "Người dự đoán không khớp nhãn",
    "DECIDED_ON_UNDETERMINED": "Có kết luận trên người có nhãn chưa xác định",
}


def _comparison_images(image, before, after, *, person_id=None, gt_rows=()) -> None:
    with st.container(key="comparison_images"):
        for column, title, result in zip(st.columns(2), (BEFORE, AFTER), (before, after)):
            with column:
                st.markdown(f"**{title}**")
                canvas = draw_result(image, result)
                for gt in gt_rows:
                    # Giới hạn tọa độ khi vẽ, kể cả nhãn ngoài phạm vi ảnh.
                    x1, x2 = [int(max(0, min(image.shape[1] - 1, float(gt[k])))) for k in ("x1", "x2")]
                    y1, y2 = [int(max(0, min(image.shape[0] - 1, float(gt[k])))) for k in ("y1", "y2")]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    text = f"GT {gt['gt_person_id']} {gt['gt_status']}"
                    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                        cv2.putText(canvas, text, (x1, max(16, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                                    .45, color, thickness, cv2.LINE_AA)
                if person_id is not None:
                    x1, y1, x2, y2 = person_view_bounds(after, person_id)
                    if x2 > x1 and y2 > y1:
                        canvas = canvas[y1:y2, x1:x2]
                st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), width="stretch")
    st.caption("Xanh lá · Đội mũ　Đỏ · Nghi không đội mũ　Cam · Chưa đủ bằng chứng")


def _show_observations(report: ComparisonReport, *, key: str) -> None:
    summary = report.summary
    suspected = str(Status.SUSPECTED_NO_HELMET)
    with st.container(key=f"{key}_metrics"):
        columns = st.columns(4)
        columns[0].metric("Người cùng được so sánh", summary["persons"])
        columns[1].metric("Nghi không mũ · Trước", summary["before_counts"][suspected])
        columns[2].metric("Nghi không mũ · Sau", summary["after_counts"][suspected])
        columns[3].metric("Người đổi kết luận", summary["changed_persons"])
    st.subheader("Nhận xét từ kết quả", anchor=False)
    for comment in report.comments:
        st.write(comment)
    st.info("Độ đúng của các kết luận được đánh giá khi có nhãn chuẩn. "
            "Số cảnh báo thay đổi thể hiện sự khác biệt giữa hai phương án.")
    st.bar_chart(pd.DataFrame({
        BEFORE: [summary["before_counts"][s] for s in STATUS_LABELS],
        AFTER: [summary["after_counts"][s] for s in STATUS_LABELS],
    }, index=list(STATUS_LABELS.values())), stack=False, color=["#648ad4", "#0f766e"])
    st.caption("Biểu đồ số người theo trạng thái kết luận.")
    with st.expander("Bảng đối chiếu từng người", expanded=True):
        changed_only = st.toggle("Chỉ xem người đổi kết luận", key=f"{key}_changed_only")
        frame = pd.DataFrame(report.rows, columns=COMPARISON_FIELDS)
        if changed_only:
            frame = frame[frame["changed"] == True]  # noqa: E712
        frame = frame.copy()
        for field in ("before_status", "after_status"):
            frame[field] = frame[field].map(STATUS_LABELS)
        st.dataframe(frame.rename(columns={
            "image_name": "Ảnh", "person_id": "Người", "before_status": "Trước",
            "after_status": "Sau", "changed": "Đổi kết luận", "before_reason": "Căn cứ trước",
            "after_rule": "Luật sau", "after_reason": "Căn cứ sau", "helmet_evidence": "Bằng chứng mũ",
            "no_helmet_evidence": "Bằng chứng đầu không mũ",
        }), hide_index=True, width="stretch")


def _show_evaluation(evaluation: dict) -> None:
    st.subheader("Đánh giá với nhãn chuẩn", anchor=False)
    old, new = evaluation["before"], evaluation["after"]
    entries = [
        ("Precision vi phạm", old["violation"]["precision"], new["violation"]["precision"]),
        ("Recall vi phạm", old["violation"]["recall"], new["violation"]["recall"]),
        ("F1 vi phạm", old["violation"]["f1"], new["violation"]["f1"]),
        ("F1 đội mũ", old["helmet"]["f1"], new["helmet"]["f1"]),
        ("Recall người theo IoU", old["person_detection_recall"], new["person_detection_recall"]),
        ("Chưa đủ bằng chứng / người khớp nhãn", old["unknown_rate_matched"], new["unknown_rate_matched"]),
    ]
    st.dataframe(pd.DataFrame([
        {"Chỉ số": label, BEFORE: "—" if a is None else f"{a:.2%}",
         AFTER: "—" if b is None else f"{b:.2%}",
         "Chênh lệch (điểm %)": "—" if a is None or b is None else f"{(b-a)*100:+.2f}"}
        for label, a, b in entries
    ]), hide_index=True, width="stretch")
    st.caption("— biểu thị chỉ số chưa xác định do mẫu số bằng 0. "
               "Recall vi phạm tính cả người bị bỏ sót và người chưa đủ bằng chứng.")
    st.bar_chart(pd.DataFrame({
        BEFORE: [a for _, a, _ in entries[:4]], AFTER: [b for _, _, b in entries[:4]],
    }, index=[label for label, _, _ in entries[:4]]), stack=False, color=["#648ad4", "#0f766e"])
    for comment in evaluation["comments"]:
        st.write(comment)
    st.caption("FP gồm cảnh báo trên người chuẩn đội mũ và cảnh báo trên người dự đoán không khớp nhãn. "
               "Người có nhãn UNDETERMINED được thống kê riêng trong ma trận nhầm lẫn.")
    sources = evaluation["gt_sources"]
    st.caption("Nguồn nhãn (số người): " + ", ".join(f"{name} ({count})" for name, count in sources.items()))
    if "derived_from_box_labels" in sources:
        st.warning("Nhãn cấp người có nguồn derived_from_box_labels được suy từ hộp chuẩn bằng quy tắc ghép "
                   "của dự án. Kết quả phụ thuộc cách sinh nhãn này; kiểm chứng độc lập quy tắc ghép cần nhãn thủ công.")
    with st.expander("Ma trận nhầm lẫn và các trường hợp cần đối chiếu"):
        for column, label, metrics in zip(st.columns(2), (BEFORE, AFTER), (old, new)):
            with column:
                st.markdown(f"**{label}**")
                cm = metrics["confusion_matrix"]
                st.dataframe(pd.DataFrame(cm["values"], index=cm["rows"], columns=cm["cols"]),
                             width="stretch")
                counts = Counter(case["error_type"] for case in metrics["cases"])
                st.dataframe(pd.DataFrame([
                    {"Loại": ERROR_LABELS.get(kind, kind), "Số trường hợp": count}
                    for kind, count in sorted(counts.items())
                ], columns=["Loại", "Số trường hợp"]), hide_index=True, width="stretch")


def _downloads(report: ComparisonReport, cfg: dict, evaluation: dict | None, *, key: str,
               errors=()) -> None:
    left, right = st.columns(2)
    frame = pd.DataFrame(report.rows, columns=COMPARISON_FIELDS)
    left.download_button("Tải bảng so sánh CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                         file_name="comparison_persons.csv", mime="text/csv", key=f"{key}_csv",
                         icon=":material/download:", width="stretch")
    payload = {**report.to_dict(), "config": cfg, "evaluation": evaluation, "errors": list(errors)}
    right.download_button("Tải báo cáo và nhận xét JSON", json.dumps(payload, ensure_ascii=False, indent=2),
                          file_name="comparison_report.json", mime="application/json", key=f"{key}_json",
                          icon=":material/download:", width="stretch")


def _single_comparison(pipeline: HelmetPipeline | None, pick_image: Callable) -> None:
    picked = pick_image("compare")
    if picked is None:
        st.info("Chọn ảnh để so sánh hai phương án.")
        return
    data, name = picked
    try:
        image = decode_image(data, name)
    except ValueError as exc:
        st.error(str(exc))
        return
    cfg = pipeline.cfg.to_dict() if pipeline is not None else None
    signature = hashlib.sha256(data + name.encode("utf-8") +
                               json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if st.button("Chạy so sánh", type="primary", icon=":material/compare:",
                 disabled=pipeline is None, key="compare_run"):
        st.session_state.pop("image_comparison", None)
        try:
            with st.spinner("Đang nhận diện và tính kết quả của hai phương án…"):
                cached = st.session_state.get("demo_analysis")
                result = cached[1] if cached is not None and cached[0] == signature else pipeline.analyze(image, name)
                report = compare_results([result])
                st.session_state.image_comparison = (signature, report)
        except Exception as exc:
            st.error(f"Chưa so sánh được ảnh. Chi tiết: {exc}")
            return
    cached = st.session_state.get("image_comparison")
    if cached is None or cached[0] != signature:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=name, width="stretch")
        st.caption("Nhấn Chạy so sánh để tạo kết quả theo cấu hình hiện tại.")
        return
    report = cached[1]
    selected = st.selectbox("Phóng to cùng một người ở hai ảnh", [None] + [p.person_id for p in report.after[0].persons],
                            format_func=lambda value: "Toàn bộ ảnh" if value is None else f"Người P{value}",
                            key=f"compare_person_{signature[:12]}")
    _comparison_images(image, report.before[0], report.after[0], person_id=selected)
    _show_observations(report, key="image_comparison")
    evaluation = None
    with st.expander("Đối chiếu ảnh này với CSV nhãn chuẩn"):
        st.caption("Các cột cần có: image_name, gt_person_id, x1, y1, x2, y2, gt_status. "
                   "Tọa độ pixel; gt_status nhận HELMET, NO_HELMET hoặc UNDETERMINED. "
                   "Tên ảnh cần khớp chính xác với ảnh đang chọn. Mỗi ảnh cần có đầy đủ nhãn cho tất cả người. "
                   "Cột source ghi nguồn nhãn nếu có.")
        upload = st.file_uploader("CSV nhãn cấp người", type=["csv"], key="compare_gt")
        if upload is not None:
            try:
                gt_rows = read_comparison_gt(upload.getvalue())
                evaluation = evaluate_comparison(report, gt_rows,
                                                 covered_images={r["image_name"] for r in gt_rows})
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Chưa đánh giá được nhãn chuẩn. {exc}")
            else:
                _show_evaluation(evaluation)
    _downloads(report, cfg, evaluation, key="image_comparison")


def _error_gallery(report: ComparisonReport, evaluation: dict, gt_rows: list[dict], image_dir: Path) -> None:
    st.subheader("Ảnh cần đối chiếu với nhãn chuẩn", anchor=False)
    method = st.radio("Trường hợp cần xem của phương án", ("before", "after"),
                      format_func=lambda value: BEFORE if value == "before" else AFTER,
                      horizontal=True, key="comparison_gallery_method")
    cases = evaluation[method]["cases"]
    if not cases:
        st.info("Không ghi nhận trường hợp cần đối chiếu trong phương án này theo nhãn đã dùng.")
        return
    kinds = sorted({c["error_type"] for c in cases})
    kind = st.selectbox("Loại trường hợp", kinds, format_func=lambda value: ERROR_LABELS.get(value, value),
                        key=f"comparison_gallery_kind_{method}")
    filtered = [c for c in cases if c["error_type"] == kind]
    index = st.selectbox("Trường hợp", range(len(filtered)),
                         format_func=lambda i: f"{filtered[i]['image_name']} · "
                         f"GT {filtered[i]['gt_person_id']} · P{filtered[i]['pred_person_id']}",
                         key=f"comparison_gallery_case_{method}_{kind}")
    case = filtered[index]
    name = case["image_name"]
    pair_index = next(i for i, result in enumerate(report.after) if result.image_name == name)
    try:
        image = read_image(image_dir / name)
    except (OSError, ValueError) as exc:
        st.warning(f"Không mở được ảnh minh họa {name}: {exc}")
        return
    _comparison_images(image, report.before[pair_index], report.after[pair_index],
                       gt_rows=[r for r in gt_rows if r["image_name"] == name])
    st.caption("Khung trắng và nhãn GT biểu thị nhãn chuẩn. Các khung màu biểu thị kết luận dự đoán.")
    st.dataframe(pd.DataFrame([case]), hide_index=True, width="stretch")


def _dataset_comparison(pipeline: HelmetPipeline | None) -> None:
    image_dir = PROJECT_ROOT / "data" / "ppe-eval" / "images" / "test"
    gt_path = PROJECT_ROOT / "data" / "person_gt" / "test_person_gt.csv"
    st.caption("Chạy mới trên bộ test có sẵn bằng cấu hình hiện tại. YOLO chạy một lần cho mỗi ảnh; "
               "hai phương án dùng chung đầu ra. Thời gian xử lý phụ thuộc thiết bị.")
    cfg = pipeline.cfg.to_dict() if pipeline is not None else None
    signature = json.dumps(cfg, sort_keys=True)
    if st.button("Chạy đánh giá trên bộ test", type="primary", disabled=pipeline is None,
                 icon=":material/analytics:", key="compare_test_run"):
        st.session_state.pop("test_comparison", None)
        progress = st.progress(0.0, text="Đang đọc bộ test…")
        try:
            paths = list_images(image_dir)
            if not paths:
                raise ValueError("Bộ test chưa có ảnh.")
            gt_rows = read_comparison_gt(gt_path.read_bytes())
            if not gt_rows:
                raise ValueError("Tệp nhãn chuẩn của bộ test đang rỗng.")
            # Chỉ giữ kết quả trong bộ nhớ phiên. Tệp trung gian được thu hồi sau lượt chạy.
            with tempfile.TemporaryDirectory(prefix="helmet_comparison_") as work_dir:
                batch = run_batch(
                    pipeline, paths, Path(work_dir), batch_size=8, save_images=False, make_charts=False,
                    progress=lambda done, total, name: progress.progress(done / total, text=f"{done}/{total} ảnh · {name}"),
                )
            if not batch.results:
                raise ValueError("Không có ảnh nào xử lý thành công để đánh giá.")
            report = compare_results(batch.results)
            evaluation = evaluate_comparison(report, gt_rows, covered_images={p.name for p in paths})
            st.session_state.test_comparison = (signature, report, evaluation, gt_rows, batch.errors)
        except Exception as exc:
            st.error(f"Chưa hoàn tất đánh giá bộ test. Chi tiết: {exc}")
        finally:
            progress.empty()
    cached = st.session_state.get("test_comparison")
    if cached is None:
        return
    if cached[0] != signature:
        st.info("Cấu hình đã thay đổi. Chạy lại đánh giá để cập nhật kết quả.")
        return
    _, report, evaluation, gt_rows, errors = cached
    st.success(f"Đã so sánh {report.summary['images']} ảnh xử lý thành công.")
    if errors:
        failed = len({e["image_path"] for e in errors})
        st.warning(f"Có {failed} ảnh gặp lỗi, được loại khỏi đánh giá. Chi tiết có trong báo cáo JSON.")
        with st.expander("Chi tiết ảnh lỗi"):
            st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")
    _show_observations(report, key="test_comparison")
    _show_evaluation(evaluation)
    _error_gallery(report, evaluation, gt_rows, image_dir)
    _downloads(report, cfg, evaluation, key="test_comparison", errors=errors)


def comparison_tab(pipeline: HelmetPipeline | None, pick_image: Callable) -> None:
    st.subheader("So sánh trước và sau hệ luật", anchor=False)
    with st.expander("Cách so sánh và đọc kết quả"):
        st.write("Trước: có tâm hộp mũ trong khung người thì gán đội mũ; thiếu mũ thì gán nghi không đội mũ. "
                 "Một mũ có thể nằm trong nhiều khung người.")
        st.write("Sau: ghép vật thể vào vùng đầu của từng người, áp dụng bằng chứng và hệ luật R1–R5, "
                 "có trạng thái chưa đủ bằng chứng.")
        st.write("Hai phương án dùng chung hộp sau lọc confidence, cùng người sau hợp nhất PPE/pose và cùng ID. "
                 "Phép so sánh đo ảnh hưởng kết hợp của bước ghép vùng đầu và hệ luật. "
                 "Nhận xét được tính trực tiếp từ kết quả trong lượt chạy này.")
    mode = st.radio("Phạm vi so sánh", ("So sánh một ảnh", "Đánh giá trên bộ test"),
                    horizontal=True, key="comparison_mode")
    if pipeline is None:
        st.warning("Mô hình PPE chưa sẵn sàng. Kiểm tra thông tin ở thanh bên.")
    if mode == "So sánh một ảnh":
        _single_comparison(pipeline, pick_image)
    else:
        _dataset_comparison(pipeline)
