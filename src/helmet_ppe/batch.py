"""Xử lý hàng loạt một thư mục ảnh và xuất toàn bộ kết quả thí nghiệm.

Cấu trúc đầu ra::

    <output_dir>/
        annotated_images/   ảnh đã vẽ kết quả (không ghi đè ảnh nguồn)
        detections.csv      từng đối tượng YOLO phát hiện
        persons.csv         từng người và trạng thái suy luận
        summary.csv         thống kê từng ảnh
        errors.csv          lỗi trong quá trình xử lý
        results.json        toàn bộ thông tin + thống kê tổng
        config.json         thông số thí nghiệm
        charts/             biểu đồ PNG
"""

from __future__ import annotations

import logging
import platform
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from helmet_ppe import charts, reporting
from helmet_ppe.config import PERSON_STATUSES
from helmet_ppe.image_io import list_images, read_image, write_image
from helmet_ppe.pipeline import HelmetPipeline
from helmet_ppe.schemas import ImageResult
from helmet_ppe.visualize import draw_result

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class BatchReport:
    output_dir: Path
    results: list[ImageResult]
    errors: list[dict]
    statistics: dict
    files: dict[str, Path] = field(default_factory=dict)


def next_experiment_dir(base: str | Path, prefix: str = "experiment") -> Path:
    """Trả về thư mục chưa tồn tại kế tiếp: experiment_01, experiment_02, ..."""
    base = Path(base)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    used = [int(m.group(1)) for p in base.glob(f"{prefix}_*")
            if p.is_dir() and (m := pattern.match(p.name))]
    return base / f"{prefix}_{max(used, default=0) + 1:02d}"


def _error(path: Path, stage: str, exc: Exception) -> dict:
    return {"image_name": path.name, "image_path": str(path), "stage": stage,
            "error_type": type(exc).__name__, "message": str(exc)}


def _chunks(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _progress_bar(total: int):
    """tqdm nếu có, ngược lại trả về None (vẫn có log theo tiến độ)."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, unit="ảnh", desc="Đang xử lý")


def _analyze_chunk(pipeline: HelmetPipeline, paths: Sequence[Path],
                   errors: list[dict]) -> list[tuple[Path, object, ImageResult]]:
    """Đọc và phân tích một lô; ảnh lỗi bị bỏ qua và ghi vào errors."""
    loaded = []
    for path in paths:
        try:
            loaded.append((path, read_image(path)))
        except Exception as exc:  # noqa: BLE001 - mọi lỗi đọc đều được ghi nhận
            errors.append(_error(path, "read", exc))
    if not loaded:
        return []
    try:
        results = pipeline.analyze_batch([img for _, img in loaded], [p for p, _ in loaded])
        return [(p, img, res) for (p, img), res in zip(loaded, results)]
    except Exception:  # noqa: BLE001 - thử lại từng ảnh để cô lập ảnh gây lỗi
        log.warning("Lô ảnh lỗi, chuyển sang xử lý từng ảnh để cô lập lỗi")
    output = []
    for path, image in loaded:
        try:
            output.append((path, image, pipeline.analyze(image, path)))
        except Exception as exc:  # noqa: BLE001
            errors.append(_error(path, "inference", exc))
    return output


def run_batch(pipeline: HelmetPipeline, image_paths: Sequence[Path], output_dir: Path, *,
              batch_size: int = 8, save_images: bool = True, make_charts: bool = True,
              progress: ProgressCallback | None = None) -> BatchReport:
    """Chạy pipeline trên danh sách ảnh và ghi đầy đủ kết quả vào ``output_dir``."""
    output_dir = Path(output_dir)
    annotated_dir = output_dir / "annotated_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(image_paths)
    total = len(image_paths)

    results: list[ImageResult] = []
    errors: list[dict] = []
    detection_rows: list[dict] = []
    person_rows: list[dict] = []
    summary_rows: list[dict] = []
    bar = _progress_bar(total)
    done = 0
    start = time.perf_counter()

    for chunk in _chunks(image_paths, max(1, batch_size)):
        for path, image, result in _analyze_chunk(pipeline, chunk, errors):
            annotated_name = ""
            if save_images:
                try:
                    write_image(annotated_dir / path.name, draw_result(image, result))
                    annotated_name = f"annotated_images/{path.name}"
                except Exception as exc:  # noqa: BLE001
                    errors.append(_error(path, "save_annotated", exc))
            results.append(result)
            detection_rows += reporting.detection_rows(result)
            person_rows += reporting.person_rows(result)
            summary_rows.append(reporting.summary_row(result, annotated_name))
        done += len(chunk)
        if bar is not None:
            bar.update(len(chunk))
        if progress is not None:
            progress(done, total, chunk[-1].name)
    if bar is not None:
        bar.close()

    wall_time = time.perf_counter() - start
    failed = len({e["image_path"] for e in errors if e["stage"] in ("read", "inference")})
    stats = reporting.compute_statistics(results, failed, wall_time)

    files = {
        "detections": reporting.write_csv(output_dir / "detections.csv", detection_rows,
                                          reporting.DETECTION_FIELDS),
        "persons": reporting.write_csv(output_dir / "persons.csv", person_rows,
                                       reporting.PERSON_FIELDS),
        "summary": reporting.write_csv(output_dir / "summary.csv", summary_rows,
                                       reporting.SUMMARY_FIELDS),
        "errors": reporting.write_csv(output_dir / "errors.csv", errors, reporting.ERROR_FIELDS),
        "config": reporting.write_json(output_dir / "config.json", {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_input_images": total,
            "batch_size": batch_size,
            "save_images": save_images,
            "python": platform.python_version(),
            "platform": platform.platform(),
            **pipeline.cfg.to_dict(),
        }),
        "results": reporting.write_json(output_dir / "results.json", {
            "statistics": stats,
            "images": [r.to_dict() for r in results],
            "errors": errors,
        }),
    }
    if make_charts:
        files.update(make_run_charts(stats, results, output_dir / "charts"))
    log.info("Hoàn tất %d/%d ảnh trong %.1f s → %s", len(results), total, wall_time, output_dir)
    return BatchReport(output_dir, results, errors, stats, files)


def run_folder(pipeline: HelmetPipeline, input_dir: str | Path, output_dir: str | Path,
               *, recursive: bool = False, **kwargs) -> BatchReport:
    """Tiện ích: liệt kê ảnh trong thư mục rồi gọi :func:`run_batch`."""
    paths = list_images(input_dir, recursive=recursive)
    if not paths:
        raise FileNotFoundError(f"Không có ảnh hợp lệ trong {input_dir}")
    return run_batch(pipeline, paths, Path(output_dir), **kwargs)


def make_run_charts(stats: dict, results: Sequence[ImageResult], out_dir: Path) -> dict[str, Path]:
    """Sinh các biểu đồ mô tả một lần chạy (không phải độ chính xác)."""
    files = {
        "chart_classes": charts.bar_chart(
            stats["detections_by_class"] or {"(không có)": 0}, out_dir / "detections_by_class.png",
            title="Số đối tượng YOLO phát hiện theo lớp", xlabel="Lớp", ylabel="Số đối tượng"),
        "chart_person_status": charts.status_bar_chart(
            {str(s): stats["person_status_counts"][str(s)] for s in PERSON_STATUSES},
            out_dir / "person_status_distribution.png",
            title="Phân bố trạng thái đội mũ (theo người)"),
        "chart_image_status": charts.status_bar_chart(
            stats["image_status_counts"], out_dir / "image_status_distribution.png",
            title="Trạng thái tổng hợp theo ảnh", ylabel="Số ảnh"),
        "chart_success": charts.bar_chart(
            {"Thành công": stats["images_processed"], "Thất bại": stats["images_failed"]},
            out_dir / "images_success_failure.png", title="Số ảnh xử lý thành công / thất bại",
            xlabel="Kết quả", ylabel="Số ảnh", colors=[charts.STATUS_COLORS["HELMET_DETECTED"],
                                                       charts.STATUS_COLORS["SUSPECTED_NO_HELMET"]]),
        "chart_time": charts.histogram(
            [r.inference_ms for r in results], out_dir / "inference_time.png",
            title="Thời gian suy luận YOLO mỗi ảnh", xlabel="ms / ảnh"),
    }
    return files


__all__ = ["BatchReport", "make_run_charts", "next_experiment_dir", "run_batch", "run_folder"]
