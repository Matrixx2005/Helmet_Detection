"""Giao diện demo Streamlit.

Chạy:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from helmet_ppe.batch import run_batch  # noqa: E402
from helmet_ppe.config import DEFAULT_OUTPUT_DIR, DetectorConfig, PipelineConfig, Status  # noqa: E402
from helmet_ppe.image_io import decode_image, list_images  # noqa: E402
from helmet_ppe.pipeline import HelmetPipeline  # noqa: E402
from helmet_ppe.reporting import person_rows, read_csv  # noqa: E402
from helmet_ppe.rule_engine import RULE_DESCRIPTIONS  # noqa: E402
from helmet_ppe.visualize import draw_result  # noqa: E402
from app.demo_views import show_demo_result, show_result_overview, show_status_metrics  # noqa: E402
from app.comparison_views import comparison_tab  # noqa: E402

STATUS_VI = {
    str(Status.HELMET_DETECTED): "Đội mũ",
    str(Status.SUSPECTED_NO_HELMET): "Nghi không đội mũ",
    str(Status.UNKNOWN): "Chưa đủ bằng chứng",
    str(Status.NO_PERSON_DETECTED): "Không có người",
}
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
SAMPLE_DIR = ROOT / "data" / "crowd_samples"

# Chế độ chia sẻ: khi chạy công khai (Streamlit Cloud), người xem không được duyệt
# thư mục trên máy chủ và số ảnh mỗi lần bị giới hạn. scripts\run_app.bat bật chế độ
# máy cá nhân bằng biến môi trường HELMET_LOCAL_MODE=1.
LOCAL_MODE = os.environ.get("HELMET_LOCAL_MODE") == "1"
MAX_PUBLIC_UPLOADS = 30

st.set_page_config(page_title="Helmet Safety · Phân tích mũ bảo hộ", page_icon="⛑️",
                   layout="wide", initial_sidebar_state="collapsed")


# --------------------------------------------------------------------------- tài nguyên
@st.cache_resource(show_spinner="Đang nạp mô hình YOLO…")
def load_pipeline(model_path: str, conf_person: float, conf_helmet: float,
                  conf_no_helmet: float, use_pose: bool, conf_pose_person: float,
                  imgsz: int) -> HelmetPipeline:
    return HelmetPipeline(PipelineConfig(detector=DetectorConfig(
        model_path=model_path, conf_person=conf_person, conf_helmet=conf_helmet,
        conf_no_helmet=conf_no_helmet, use_pose=use_pose, conf_pose_person=conf_pose_person,
        imgsz=imgsz)))


def to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def zip_folder(folder: Path, patterns: tuple[str, ...] = ("*",)) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for pattern in patterns:
            for path in sorted(folder.rglob(pattern)):
                if path.is_file():
                    archive.write(path, path.relative_to(folder))
    return buffer.getvalue()


def persons_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for column in ("person_confidence", "helmet_confidence", "helmet_match_score",
                   "helmet_evidence", "no_helmet_confidence", "no_helmet_match_score",
                   "no_helmet_evidence"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Trạng thái"] = df["status"].map(STATUS_VI)
    cols = {"image_name": "Ảnh", "person_id": "Person ID", "Trạng thái": "Trạng thái",
            "rule_id": "Luật", "source": "Nguồn", "person_confidence": "Conf. người",
            "helmet_confidence": "Conf. mũ", "helmet_match_score": "Điểm ghép mũ",
            "helmet_evidence": "Bằng chứng mũ", "no_helmet_confidence": "Conf. no_helmet",
            "no_helmet_match_score": "Điểm ghép no_helmet",
            "no_helmet_evidence": "Bằng chứng no_helmet", "reason": "Giải thích"}
    return df[[c for c in cols if c in df.columns]].rename(columns=cols)


# --------------------------------------------------------------------------- sidebar
def sidebar() -> HelmetPipeline | None:
    st.sidebar.markdown("## ⛑ Helmet Safety")
    st.sidebar.caption("Điều chỉnh cấu hình cho ảnh công trường của bạn.")
    settings = st.sidebar.expander("Cấu hình phân tích", expanded=False)
    defaults = DetectorConfig()
    model = Path(defaults.model_path)
    if not model.is_file():
        model = ROOT / "models" / "yolo26n.pt"
    if not model.is_file():
        st.sidebar.error("Thiếu mô hình PPE trong thư mục models/.")
        return None
    settings.markdown("**Ngưỡng confidence phát hiện**")
    conf_person = settings.slider("Person", 0.05, 0.95, defaults.conf_person, 0.05)
    conf_helmet = settings.slider("helmet", 0.05, 0.95, defaults.conf_helmet, 0.05)
    conf_no_helmet = settings.slider("no_helmet", 0.01, 0.50, defaults.conf_no_helmet, 0.01)
    pose_available = Path(defaults.pose_model_path).is_file()
    use_pose = settings.toggle("Dùng mô hình pose (ảnh đông người)",
                                 value=defaults.use_pose and pose_available,
                                 disabled=not pose_available,
                                 help="YOLO26n-pose bổ sung người bị che khuất và định vị đầu thật "
                                      "để không gán nhầm mũ của người đứng phía sau.")
    conf_pose_person = settings.slider("Người (pose)", 0.05, 0.95, defaults.conf_pose_person,
                                         0.05, disabled=not use_pose)
    imgsz = settings.select_slider("Kích thước suy luận", [320, 416, 480, 640, 800, 960, 1280],
                                     value=defaults.imgsz)
    settings.caption("Giá trị mặc định được dò trên tập validation. Ngưỡng thấp → phát hiện "
                       "nhiều hơn nhưng dễ báo nhầm. Lớp no_helmet hiếm nên mô hình cho "
                       "confidence thấp, cần ngưỡng riêng nhỏ hơn.")
    try:
        return load_pipeline(str(model), conf_person, conf_helmet, conf_no_helmet,
                             use_pose, conf_pose_person, imgsz)
    except Exception as exc:  # hiển thị lỗi thay vì làm sập ứng dụng
        st.sidebar.error(f"Không nạp được mô hình: {exc}")
        return None


# --------------------------------------------------------------------------- chế độ một ảnh
def _pick_single_image(key_prefix: str = "single") -> tuple[bytes, str] | None:
    """Ảnh người dùng tải lên, hoặc một ảnh mẫu có sẵn trong dự án."""
    samples = list_images(SAMPLE_DIR) if SAMPLE_DIR.is_dir() else []
    source = "Tải ảnh lên"
    if samples:
        source = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Dùng ảnh mẫu (đám đông)"], index=1,
                          horizontal=True, key=f"{key_prefix}_source")
    if source == "Tải ảnh lên":
        upload = st.file_uploader("Chọn một ảnh", type=UPLOAD_TYPES, key=key_prefix)
        return (upload.getvalue(), upload.name) if upload is not None else None
    sample = st.selectbox("Ảnh mẫu", samples, format_func=lambda p: p.name, key=f"{key_prefix}_sample")
    try:
        return sample.read_bytes(), sample.name
    except OSError as exc:
        st.error(f"Không đọc được ảnh mẫu {sample.name}: {exc}")
        return None


def single_image_tab(pipeline: HelmetPipeline | None) -> None:
    preview, controls = st.columns([2.3, 1], gap="large")
    display = preview.container(key="single_preview")
    with controls, st.container(border=True, key="input_panel"):
        st.subheader("Chọn ảnh phân tích", anchor=False)
        picked = _pick_single_image()
        action = st.empty()
        st.caption("Chọn ảnh rõ vùng đầu để kiểm tra tình trạng đội mũ của từng người.")
        if pipeline is not None:
            st.caption("Hỗ trợ ảnh đông người bằng pose" if pipeline.pose is not None
                       else "Đang sử dụng mô hình phát hiện mũ bảo hộ")
        else:
            st.error("Chưa có mô hình PPE sẵn sàng. Mở cấu hình ở thanh bên để xem chi tiết.")
    if picked is None:
        with display, st.container(border=True):
            st.subheader("Phân tích mũ bảo hộ")
            st.info("Tải ảnh công trường ở bên cạnh để bắt đầu.")
        return
    data, name = picked
    try:
        image = decode_image(data, name)
    except ValueError as exc:
        st.error(str(exc))
        return

    config = pipeline.cfg.to_dict() if pipeline is not None else None
    signature = hashlib.sha256(data + name.encode("utf-8") +
                               json.dumps(config, sort_keys=True).encode()).hexdigest()
    cached = st.session_state.get("demo_analysis")
    has_result = cached is not None and cached[0] == signature
    run = action.button("Phân tích lại" if has_result else "Phân tích ảnh", type="primary",
                        icon=":material/document_scanner:", width="stretch",
                        disabled=pipeline is None)
    if run:
        try:
            with st.spinner("Đang nhận diện, ghép vùng đầu và suy luận hệ luật…"):
                result = pipeline.analyze(image, name)
        except Exception as exc:
            st.error(f"Phân tích ảnh thất bại. Có thể thử lại. Chi tiết lỗi: {exc}")
            return
        st.session_state.demo_analysis = (signature, result)
        st.session_state.demo_step = 3
        has_result = True
    if not has_result:
        st.session_state.demo_step = 0
        with display:
            st.image(to_rgb(image), caption=f"{name} · {image.shape[1]} × {image.shape[0]} px",
                     width="stretch")
        return
    result = st.session_state.demo_analysis[1]
    with display:
        show_status_metrics(len(result.persons), {
            str(status): result.count_status(status) for status in
            (Status.HELMET_DETECTED, Status.SUSPECTED_NO_HELMET, Status.UNKNOWN)
        }, key="single_metrics")
        show_result_overview(image, result, pipeline.cfg, signature[:12], controls)
    st.caption(f"Thời gian phân tích {result.total_ms:.0f} ms · "
               f"Nhận diện {result.inference_ms:.0f} ms")
    with st.container(key="result_downloads"):
        downloads = st.columns(3)
    success, encoded = cv2.imencode(".png", draw_result(image, result))
    if success:
        downloads[0].download_button("Tải ảnh kết quả", encoded.tobytes(),
                                     file_name=f"{Path(name).stem}_result.png", mime="image/png",
                                     icon=":material/download:", width="stretch")
    table = persons_table(person_rows(result))
    downloads[1].download_button("Tải bảng CSV", table.to_csv(index=False).encode("utf-8-sig"),
                                 file_name=f"{Path(name).stem}_persons.csv", mime="text/csv",
                                 icon=":material/table:", width="stretch")
    downloads[2].download_button("Tải kết quả JSON",
                                 json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                                 file_name=f"{Path(name).stem}_results.json", mime="application/json",
                                 icon=":material/data_object:", width="stretch")
    with st.expander("Xem cách hệ thống kết luận"):
        show_demo_result(image, result, pipeline.cfg, signature[:12])
    with st.expander("Bảng kết quả từng người"):
        if not table.empty:
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.info("Không phát hiện người để lập bảng kết quả.")
    with st.expander("Dữ liệu kỹ thuật JSON"):
        st.json(result.to_dict())


# --------------------------------------------------------------------------- chế độ nhiều ảnh
def _collect_inputs() -> tuple[list[Path], list]:
    """Trả về (ảnh trong thư mục có sẵn, ảnh tải lên) — chỉ một trong hai khác rỗng."""
    sources = ["Tải lên nhiều ảnh"] + (["Thư mục có sẵn trên máy"] if LOCAL_MODE else [])
    source = st.radio("Nguồn ảnh", sources, horizontal=True)
    if source == "Tải lên nhiều ảnh":
        uploads = list(st.file_uploader("Chọn nhiều ảnh", type=UPLOAD_TYPES,
                                        accept_multiple_files=True, key="multi") or [])
        if not LOCAL_MODE and len(uploads) > MAX_PUBLIC_UPLOADS:
            st.warning(f"Bản chia sẻ chỉ xử lý tối đa {MAX_PUBLIC_UPLOADS} ảnh mỗi lần; "
                       f"đang lấy {MAX_PUBLIC_UPLOADS} ảnh đầu tiên.")
            uploads = uploads[:MAX_PUBLIC_UPLOADS]
        return [], uploads

    folder = st.text_input("Đường dẫn thư mục ảnh", str(ROOT / "data" / "test_images"))
    try:
        paths = list_images(folder)
    except NotADirectoryError as exc:
        st.warning(str(exc))
        return [], []
    st.caption(f"Tìm thấy {len(paths)} ảnh hợp lệ.")
    return paths, []


def batch_tab(pipeline: HelmetPipeline) -> None:
    st.subheader("Phân tích nhiều ảnh")
    st.caption("Tải ảnh công trường để tổng hợp người cần kiểm tra và xuất báo cáo.")
    paths, uploads = _collect_inputs()
    st.caption(f"{len(paths) + len(uploads)} ảnh đã chọn")
    run = st.button("Phân tích hàng loạt", type="primary", icon=":material/play_arrow:",
                    disabled=not (paths or uploads))
    if not run:
        _show_batch_result()
        return

    if LOCAL_MODE:
        out_dir = DEFAULT_OUTPUT_DIR / f"streamlit_{datetime.now():%Y%m%d_%H%M%S}"
    else:  # bản chia sẻ: mỗi lượt chạy một thư mục tạm riêng
        out_dir = Path(tempfile.mkdtemp(prefix="helmet_batch_"))
    if uploads:  # lưu bản sao ảnh tải lên, không đụng tới ảnh gốc của người dùng
        input_dir = out_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        for upload in uploads:
            (input_dir / Path(upload.name).name).write_bytes(upload.getvalue())
        paths = list_images(input_dir)

    bar = st.progress(0.0, text="Đang xử lý…")

    def on_progress(done: int, total: int, name: str) -> None:
        bar.progress(done / total, text=f"{done}/{total} ảnh · {name}")

    try:
        with st.spinner("Đang phân tích ảnh và tạo báo cáo…"):
            report = run_batch(pipeline, paths, out_dir, progress=on_progress)
    except Exception as exc:
        bar.empty()
        st.error(f"Xử lý hàng loạt thất bại. Có thể thử lại. Chi tiết lỗi: {exc}")
        return
    bar.progress(1.0, text="Hoàn tất")
    st.session_state["batch_dir"] = str(report.output_dir)
    _show_batch_result()


def _show_batch_result() -> None:
    if "batch_dir" not in st.session_state:
        return
    out_dir = Path(st.session_state["batch_dir"])
    stats = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))["statistics"]
    counts = stats["person_status_counts"]

    st.success(f"Kết quả lưu tại: {out_dir}" if LOCAL_MODE
               else "Hoàn tất. Tải kết quả ở thẻ \"Tải về\".")
    show_status_metrics(stats["persons_total"], counts, key="batch_metrics")
    if stats["images_failed"]:
        st.warning(f"{stats['images_failed']} ảnh xử lý lỗi. Xem chi tiết trong tệp errors.csv.")
    st.caption(f"{stats['images_processed']} ảnh thành công · {stats['images_failed']} ảnh lỗi · "
               f"Tổng thời gian {stats['wall_time_s']:.1f} s · trung bình "
               f"{stats['avg_ms_per_image'] or 0:.0f} ms/ảnh")

    summary = pd.DataFrame(read_csv(out_dir / "summary.csv"))
    tab_sum, tab_person, tab_img, tab_chart, tab_dl = st.tabs(
        ["Tổng hợp theo ảnh", "Theo người", "Ảnh kết quả", "Biểu đồ", "Tải về"])
    with tab_sum:
        if not summary.empty:
            only_suspected = st.toggle("Chỉ hiện ảnh có người nghi không đội mũ")
            if only_suspected:
                summary = summary[summary["image_status"] == str(Status.SUSPECTED_NO_HELMET)]
            summary["image_status"] = summary["image_status"].map(STATUS_VI)
        st.dataframe(summary.drop(columns=["annotated_image"], errors="ignore"),
                     width="stretch", hide_index=True)
    with tab_person:
        st.dataframe(persons_table(read_csv(out_dir / "persons.csv")), width="stretch",
                     hide_index=True)
    with tab_img:
        images = sorted((out_dir / "annotated_images").glob("*"))
        if images:
            chosen = st.selectbox("Chọn ảnh", images, format_func=lambda p: p.name)
            st.image(str(chosen), width="stretch")
    with tab_chart:
        cols = st.columns(2)
        for i, chart in enumerate(sorted((out_dir / "charts").glob("*.png"))):
            cols[i % 2].image(str(chart), width="stretch")
    with tab_dl:
        for name, mime in (("summary.csv", "text/csv"), ("persons.csv", "text/csv"),
                           ("detections.csv", "text/csv"), ("results.json", "application/json"),
                           ("errors.csv", "text/csv")):
            st.download_button(name, (out_dir / name).read_bytes(), file_name=name, mime=mime,
                               icon=":material/download:")
        st.download_button("⬇️ Ảnh kết quả (ZIP)", zip_folder(out_dir / "annotated_images"),
                           file_name="annotated_images.zip", mime="application/zip")


# --------------------------------------------------------------------------- trang chính
def about_tab() -> None:
    st.markdown(
        "**Luồng xử lý:** Ảnh → YOLO26n (Person, helmet, no_helmet) → ghép mũ với vùng đầu "
        "từng người → hệ luật → kết luận.\n\n"
        "Hệ thống **không** coi việc không thấy mũ là vi phạm chắc chắn: chỉ khi mô hình phát "
        "hiện vùng đầu không đội mũ với bằng chứng đủ mạnh mới kết luận *nghi không đội mũ*.")
    st.table(pd.DataFrame({"Luật": list(RULE_DESCRIPTIONS),
                           "Ý nghĩa": list(RULE_DESCRIPTIONS.values())}))


def main() -> None:
    st.html(f"<style>{(ROOT / 'app' / 'style.css').read_text(encoding='utf-8')}</style>")
    pipeline = sidebar()
    with st.container(key="dashboard_header"):
        heading, status = st.columns([2.3, 1], gap="large")
        with heading:
            st.caption("HELMET SAFETY")
            st.title("An toàn mũ bảo hộ")
            st.write("Phân tích ảnh công trường và kiểm tra bằng chứng của từng người.")
        status_key = "system_status" if pipeline is not None else "system_status_unavailable"
        with status, st.container(key=status_key):
            if pipeline is None:
                st.markdown("**Hệ thống cần kiểm tra**")
                st.caption("Xem thông tin mô hình ở thanh bên.")
            else:
                st.markdown("**Sẵn sàng phân tích**")
                st.caption("Pose đang hỗ trợ ảnh đông người" if pipeline.pose is not None
                           else "Nhận diện người, mũ và vùng đầu")
    single, batch, comparison, about = st.tabs(
        ["Phân tích ảnh", "Xử lý hàng loạt", "So sánh & đánh giá", "Giải thích hệ thống"])
    with about:
        about_tab()
    with single:
        single_image_tab(pipeline)
    with comparison:
        comparison_tab(pipeline, _pick_single_image)
    with batch:
        if pipeline is not None:
            batch_tab(pipeline)
        else:
            st.info("Mô hình PPE chưa sẵn sàng để xử lý hàng loạt. Xem chi tiết ở thanh bên.")


main()
