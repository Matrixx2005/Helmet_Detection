"""Giao diện demo Streamlit.

Chạy:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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

st.set_page_config(page_title="Phát hiện mũ bảo hộ", page_icon="⛑️", layout="wide")


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
    df["Trạng thái"] = df["status"].map(STATUS_VI)
    cols = {"image_name": "Ảnh", "person_id": "Person ID", "Trạng thái": "Trạng thái",
            "rule_id": "Luật", "source": "Nguồn", "person_confidence": "Conf. người",
            "helmet_confidence": "Conf. mũ", "helmet_match_score": "Điểm ghép mũ",
            "no_helmet_confidence": "Conf. no_helmet", "reason": "Giải thích"}
    return df[[c for c in cols if c in df.columns]].rename(columns=cols)


# --------------------------------------------------------------------------- sidebar
def sidebar() -> HelmetPipeline | None:
    st.sidebar.header("⚙️ Cài đặt")
    models = sorted(p for p in (ROOT / "models").glob("*.pt") if "pose" not in p.stem)
    if not models:
        st.sidebar.error("Không có mô hình .pt trong thư mục models/")
        return None
    default = next((i for i, m in enumerate(models) if m.name == "best.pt"), 0)
    model = st.sidebar.selectbox("Mô hình", models, index=default, format_func=lambda p: p.name)
    defaults = DetectorConfig()
    st.sidebar.markdown("**Confidence threshold**")
    conf_person = st.sidebar.slider("Person", 0.05, 0.95, defaults.conf_person, 0.05)
    conf_helmet = st.sidebar.slider("helmet", 0.05, 0.95, defaults.conf_helmet, 0.05)
    conf_no_helmet = st.sidebar.slider("no_helmet", 0.01, 0.50, defaults.conf_no_helmet, 0.01)
    pose_available = Path(defaults.pose_model_path).is_file()
    use_pose = st.sidebar.toggle("Dùng mô hình pose (ảnh đông người)",
                                 value=defaults.use_pose and pose_available,
                                 disabled=not pose_available,
                                 help="YOLO26n-pose bổ sung người bị che khuất và định vị đầu thật "
                                      "để không gán nhầm mũ của người đứng phía sau.")
    conf_pose_person = st.sidebar.slider("Người (pose)", 0.05, 0.95, defaults.conf_pose_person,
                                         0.05, disabled=not use_pose)
    imgsz = st.sidebar.select_slider("Image size", [320, 416, 480, 640, 800, 960, 1280],
                                     value=defaults.imgsz)
    st.sidebar.caption("Giá trị mặc định được dò trên tập validation. Ngưỡng thấp → phát hiện "
                       "nhiều hơn nhưng dễ báo nhầm. Lớp no_helmet hiếm nên mô hình cho "
                       "confidence thấp, cần ngưỡng riêng nhỏ hơn.")
    try:
        return load_pipeline(str(model), conf_person, conf_helmet, conf_no_helmet,
                             use_pose, conf_pose_person, imgsz)
    except Exception as exc:  # hiển thị lỗi thay vì làm sập ứng dụng
        st.sidebar.error(f"Không nạp được mô hình: {exc}")
        return None


# --------------------------------------------------------------------------- chế độ một ảnh
def _pick_single_image() -> tuple[bytes, str] | None:
    """Ảnh người dùng tải lên, hoặc một ảnh mẫu có sẵn trong dự án."""
    samples = list_images(SAMPLE_DIR) if SAMPLE_DIR.is_dir() else []
    source = "Tải ảnh lên"
    if samples:
        source = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Dùng ảnh mẫu (đám đông)"],
                          horizontal=True, key="single_source")
    if source == "Tải ảnh lên":
        upload = st.file_uploader("Chọn một ảnh", type=UPLOAD_TYPES, key="single")
        return (upload.getvalue(), upload.name) if upload is not None else None
    sample = st.selectbox("Ảnh mẫu", samples, format_func=lambda p: p.name)
    return sample.read_bytes(), sample.name


def single_image_tab(pipeline: HelmetPipeline) -> None:
    picked = _pick_single_image()
    if picked is None:
        st.info("Tải lên một ảnh công trường để bắt đầu.")
        return
    data, name = picked
    try:
        image = decode_image(data, name)
    except ValueError as exc:
        st.error(str(exc))
        return

    left, right = st.columns(2)
    left.image(to_rgb(image), caption="Ảnh gốc", width="stretch")
    if not st.button("🔍 Phân tích", type="primary"):
        return

    result = pipeline.analyze(image, name)
    right.image(to_rgb(draw_result(image, result)), caption="Kết quả nhận diện", width="stretch")

    c = st.columns(5)
    c[0].metric("Số người", len(result.persons))
    c[1].metric("Số mũ phát hiện", result.count("helmet"))
    c[2].metric("Đội mũ", result.count_status(Status.HELMET_DETECTED))
    c[3].metric("Nghi không đội mũ", result.count_status(Status.SUSPECTED_NO_HELMET))
    c[4].metric("Chưa đủ bằng chứng", result.count_status(Status.UNKNOWN))
    st.markdown(f"**Trạng thái ảnh:** {STATUS_VI[str(result.image_status)]} "
                f"· thời gian suy luận {result.inference_ms:.0f} ms")

    if result.persons:
        st.subheader("Phân tích từng người")
        st.dataframe(persons_table(person_rows(result)), width="stretch", hide_index=True)
    with st.expander("Dữ liệu JSON đầy đủ"):
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
    paths, uploads = _collect_inputs()
    if not (paths or uploads) or not st.button("▶️ Bắt đầu phân tích", type="primary"):
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

    report = run_batch(pipeline, paths, out_dir, progress=on_progress)
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
    c = st.columns(6)
    c[0].metric("Ảnh thành công", stats["images_processed"])
    c[1].metric("Ảnh lỗi", stats["images_failed"])
    c[2].metric("Tổng số người", stats["persons_total"])
    c[3].metric("Đội mũ", counts[str(Status.HELMET_DETECTED)])
    c[4].metric("Nghi không đội mũ", counts[str(Status.SUSPECTED_NO_HELMET)])
    c[5].metric("Chưa đủ bằng chứng", counts[str(Status.UNKNOWN)])
    st.caption(f"Tổng thời gian {stats['wall_time_s']:.1f} s · trung bình "
               f"{stats['avg_ms_per_image'] or 0:.0f} ms/ảnh")

    summary = pd.DataFrame(read_csv(out_dir / "summary.csv"))
    tab_sum, tab_person, tab_img, tab_chart, tab_dl = st.tabs(
        ["Tổng hợp theo ảnh", "Theo người", "Ảnh kết quả", "Biểu đồ", "Tải về"])
    with tab_sum:
        if not summary.empty:
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
                           ("detections.csv", "text/csv"), ("results.json", "application/json")):
            st.download_button(f"⬇️ {name}", (out_dir / name).read_bytes(), file_name=name, mime=mime)
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
    st.title("⛑️ Phát hiện mũ bảo hộ lao động & suy luận vi phạm")
    pipeline = sidebar()
    single, batch, about = st.tabs(["🖼️ Một ảnh", "🗂️ Nhiều ảnh", "ℹ️ Phương pháp"])
    with about:
        about_tab()
    if pipeline is None:
        st.stop()
    with single:
        single_image_tab(pipeline)
    with batch:
        batch_tab(pipeline)


main()
