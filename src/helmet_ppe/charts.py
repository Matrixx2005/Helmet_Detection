"""Biểu đồ PNG phục vụ báo cáo (matplotlib, không cần màn hình)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (phải đặt backend trước khi import)
import numpy as np  # noqa: E402

# Bảng màu: một màu cho chuỗi đơn, màu trạng thái cố định, chữ màu trung tính.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
STATUS_COLORS = {
    "HELMET_DETECTED": "#0ca30c",
    "SUSPECTED_NO_HELMET": "#d03b3b",
    "UNKNOWN": "#fab219",
    "NO_PERSON_DETECTED": "#8c8b86",
    "MISSED": "#8c8b86",
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=TEXT, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def bar_chart(values: Mapping[str, float], path: Path, *, title: str, xlabel: str,
              ylabel: str, colors: Sequence[str] | None = None,
              value_fmt: str = "{:.0f}", ylim: tuple[float, float] | None = None) -> Path:
    """Biểu đồ cột một chuỗi, ghi giá trị trên đầu mỗi cột."""
    labels = list(values)
    data = [float(values[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * len(labels) + 2), 4))
    bars = ax.bar(labels, data, color=colors or SERIES[0], width=0.6,
                  edgecolor="white", linewidth=2)
    for bar, value in zip(bars, data):
        ax.annotate(value_fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=TEXT)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, max(data + [1]) * 1.15)
    _style(ax, title, xlabel, ylabel)
    if len(labels) > 4:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path)


def status_bar_chart(counts: Mapping[str, int], path: Path, *, title: str,
                     xlabel: str = "Trạng thái", ylabel: str = "Số lượng") -> Path:
    """Cột theo trạng thái, dùng màu trạng thái cố định (luôn kèm nhãn chữ)."""
    colors = [STATUS_COLORS.get(k, SERIES[0]) for k in counts]
    return bar_chart(counts, path, title=title, xlabel=xlabel, ylabel=ylabel, colors=colors)


def grouped_bar_chart(groups: Sequence[str], series: Mapping[str, Sequence[float]], path: Path,
                      *, title: str, xlabel: str, ylabel: str,
                      ylim: tuple[float, float] = (0, 1.05), value_fmt: str = "{:.2f}") -> Path:
    """Cột nhóm (ví dụ Precision/Recall/mAP theo lớp), có chú thích."""
    n_series = len(series)
    x = np.arange(len(groups))
    bar_w = 0.8 / max(n_series, 1)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(groups) + 2), 4.2))
    for i, (name, values) in enumerate(series.items()):
        offsets = x - 0.4 + bar_w * (i + 0.5)
        bars = ax.bar(offsets, values, bar_w, label=name, color=SERIES[i % len(SERIES)],
                      edgecolor="white", linewidth=1.5)
        for bar, value in zip(bars, values):
            ax.annotate(value_fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=TEXT)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncol=min(n_series, 4), loc="upper left",
              bbox_to_anchor=(0, -0.12))
    _style(ax, title, xlabel, ylabel)
    return _save(fig, path)


def histogram(values: Sequence[float], path: Path, *, title: str, xlabel: str,
              ylabel: str = "Số ảnh") -> Path:
    """Histogram kèm đường trung bình (dùng cho thời gian xử lý)."""
    data = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if data.size:
        ax.hist(data, bins=min(30, max(5, data.size // 3)), color=SERIES[0],
                edgecolor="white", linewidth=1)
        mean = float(data.mean())
        ax.axvline(mean, color=TEXT, linestyle="--", linewidth=1.2)
        ax.annotate(f"TB = {mean:.1f}", (mean, ax.get_ylim()[1] * 0.95), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=TEXT, va="top")
    _style(ax, title, xlabel, ylabel)
    return _save(fig, path)


def confusion_matrix_chart(matrix: Sequence[Sequence[int]], row_labels: Sequence[str],
                           col_labels: Sequence[str], path: Path, *, title: str,
                           xlabel: str = "Dự đoán", ylabel: str = "Nhãn chuẩn") -> Path:
    """Ma trận nhầm lẫn (thang xanh một sắc độ, ghi số trong ô)."""
    data = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 * len(col_labels) + 2.5, 1.0 * len(row_labels) + 2))
    ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=max(1.0, data.max()))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dark = data[i, j] > 0.55 * max(1.0, data.max())
            ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center",
                    color="white" if dark else TEXT, fontsize=11)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=15, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, fontsize=12, color=TEXT, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, path)
