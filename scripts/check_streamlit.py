"""Kiểm tra giao diện Streamlit: chạy script ở chế độ test và khởi động server thật.

    python scripts/check_streamlit.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
PORT = 8599


def check_app_script() -> None:
    """Chạy toàn bộ script bằng AppTest (bao gồm nạp mô hình) và đảm bảo không có lỗi."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    if app.exception:
        raise RuntimeError(f"Ứng dụng phát sinh lỗi: {[e.message for e in app.exception]}")
    print(f"[OK] AppTest: tiêu đề = {app.title[0].value!r}, số tab = {len(app.tabs)}")


def check_sample_analysis() -> None:
    """Bản chia sẻ: chọn ảnh mẫu, bấm Phân tích, kiểm tra có kết quả và không lộ thư mục máy chủ."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    sources = [r for r in app.radio if r.key == "single_source"]
    if not sources:
        raise RuntimeError("Không thấy lựa chọn ảnh mẫu (thiếu data/crowd_samples?)")
    sources[0].set_value("Dùng ảnh mẫu (đám đông)").run()
    next(b for b in app.button if "Phân tích" in b.label).click().run()
    if app.exception:
        raise RuntimeError(f"Lỗi khi phân tích ảnh mẫu: {[e.message for e in app.exception]}")
    metrics = {m.label: m.value for m in app.metric}
    if "Số người" not in metrics:
        raise RuntimeError("Không có kết quả phân tích ảnh mẫu")
    batch_sources = [o for r in app.radio if r.key != "single_source" for o in r.options]
    if any("Thư mục" in o for o in batch_sources):
        raise RuntimeError("Bản chia sẻ không được cho chọn thư mục trên máy chủ")
    print(f"[OK] Ảnh mẫu: {metrics}")


def check_server() -> None:
    """Khởi động server headless, gọi endpoint health rồi tắt."""
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true",
           "--server.port", str(PORT), "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2) as r:
                    print(f"[OK] Server Streamlit phản hồi: {r.read().decode()} (cổng {PORT})")
                    return
            except OSError:
                continue
        raise RuntimeError("Server Streamlit không phản hồi sau 60 giây")
    finally:
        proc.terminate()
        proc.wait(timeout=20)


if __name__ == "__main__":
    check_app_script()
    check_sample_analysis()
    check_server()
