# ⛑️ Phát hiện mũ bảo hộ lao động & suy luận vi phạm bằng hệ luật

Đồ án môn Học sâu. Hệ thống nhận ảnh công trường, phát hiện **người**, **mũ bảo
hộ** và **vùng đầu không đội mũ** bằng YOLO26n, ghép mũ với đúng người bằng thuật
toán hình học, rồi dùng **hệ luật** để kết luận từng người:
`HELMET_DETECTED`, `SUSPECTED_NO_HELMET`, `UNKNOWN` hoặc `NO_PERSON_DETECTED`.

```
Ảnh → YOLO26n → Association (ghép mũ ↔ người) → Rule Engine → ảnh kết quả + CSV/JSON + biểu đồ
```

Kết quả chính và phân tích: [docs/06_results.md](docs/06_results.md).

## 1. Cài đặt

Yêu cầu: Windows 10/11, Python 3.10+ (đã thử với 3.13.11), khoảng 2 GB trống.

```powershell
cd F:\helmet-violation-detection
python -m venv .venv                       # bỏ qua nếu đã có .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Hoặc nhấp đúp **`scripts\setup.bat`** (cài thư viện, chuyển code thử nghiệm cũ vào
`legacy\`, tạo commit Git đầu tiên nếu máy có Git).

Mô hình: `models\best.pt` (bắt buộc, mô hình chính) và `models\best_v2.pt`
(tùy chọn, mô hình đối chứng).

> Mọi lệnh bên dưới chạy từ thư mục gốc dự án và cần `PYTHONPATH=src`. Trong
> PowerShell: `$env:PYTHONPATH="src"`; trong cmd: `set PYTHONPATH=src`. Các tệp
> `.bat` trong `scripts\` đã tự đặt sẵn.

## 2. Chạy một ảnh

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m helmet_ppe predict "data\raw\anh_ct_nh (1).jpg"
```

In trạng thái từng người ra màn hình và lưu ảnh đã vẽ + JSON vào `outputs\single\`.

## 3. Xử lý hàng loạt (batch)

```powershell
.venv\Scripts\python.exe -m helmet_ppe batch --input data\test_images
# hoặc nhấp đúp scripts\run_batch.bat   /   scripts\run_batch.bat <thư_mục_ảnh> <thư_mục_kết_quả>
```

Tùy chọn hay dùng:

| Tùy chọn | Ý nghĩa | Mặc định |
|---|---|---|
| `--input` | Thư mục ảnh (JPG, JPEG, PNG, WEBP, BMP) | `data\test_images` |
| `--output` | Thư mục kết quả | `outputs\experiment_XX` mới |
| `--model` | Tệp trọng số | `models\best.pt` |
| `--conf-person` / `--conf-helmet` / `--conf-no-helmet` | Ngưỡng confidence theo lớp | 0,40 / 0,25 / 0,03 (dò trên val) |
| `--imgsz` | Kích thước ảnh vào | 640 |
| `--batch-size` | Số ảnh mỗi lô | 8 |
| `--device` | `cpu`, `0`… | tự chọn |
| `--params` | Nạp toàn bộ ngưỡng từ `best_params.json` | — |
| `--recursive`, `--no-images` | Quét thư mục con / không lưu ảnh vẽ | — |

Kết quả:

```
outputs\experiment_01\
├── annotated_images\   ảnh đã vẽ (ảnh nguồn không bị sửa)
├── detections.csv      từng đối tượng YOLO + vai trò sau khi ghép
├── persons.csv         từng người: trạng thái, luật, bằng chứng, lý do
├── summary.csv         thống kê từng ảnh
├── results.json        toàn bộ dữ liệu + thống kê tổng
├── errors.csv          ảnh lỗi (một ảnh lỗi không làm dừng cả quá trình)
├── config.json         thông số thí nghiệm
└── charts\             biểu đồ PNG
```

Màu trên ảnh: **xanh lá** = đội mũ, **đỏ** = nghi không đội mũ, **cam** = chưa đủ
bằng chứng, **xám** = mũ / no_helmet không thuộc người nào. Nhãn `P2 HELMET 0.87`
= người số 2, trạng thái, độ tin cậy của bằng chứng.

## 4. Giao diện Streamlit

```powershell
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
# hoặc nhấp đúp scripts\run_app.bat
```

Mở <http://localhost:8501>. Ba tab: **Một ảnh** (tải ảnh → Phân tích → ảnh kết
quả, số liệu, bảng từng người), **Nhiều ảnh** (tải nhiều ảnh hoặc nhập đường dẫn
thư mục → tiến độ → bảng, ảnh kết quả, biểu đồ, tải CSV/JSON/ZIP), **Phương
pháp** (bảng luật). Thanh bên: chọn mô hình, ngưỡng confidence từng lớp, image size.

## 5. Đánh giá

Cần tệp `construction-ppe.zip` (mặc định tìm ở `%USERPROFILE%\Downloads`).

```powershell
scripts\run_evaluation.bat                       # hoặc: scripts\run_evaluation.bat D:\duong\dan\construction-ppe.zip
```

Script thực hiện:

1. `prepare-dataset` – giải nén vào `data\construction-ppe\`, tạo `data_local.yaml`.
2. `eval-detector` – Precision, Recall, mAP50, mAP50-95, confusion matrix trên val
   và test → `reports\detection_eval\`.
3. `make-person-gt` – nhãn chuẩn cấp người → `data\person_gt\`.
4. `tune-rules` – dò ngưỡng trên **val** → `reports\rule_tuning\<model>\best_params.json`.
5. `batch --params …` + `eval-rules` trên **test** → `reports\rule_eval\<model>\`
   (`rule_metrics.json`, `error_cases.csv`, `error_examples\`, biểu đồ).

Gọi lẻ từng bước: `python -m helmet_ppe <lệnh> --help`.

Tự gán nhãn cấp người cho ảnh riêng: tạo CSV với các cột
`image_name, gt_person_id, x1, y1, x2, y2, gt_status, source`
(`gt_status` ∈ `HELMET`, `NO_HELMET`, `UNDETERMINED`) rồi chạy
`python -m helmet_ppe eval-rules --run outputs\experiment_01 --gt <file.csv>`.

## 6. Kiểm thử

```powershell
.venv\Scripts\python.exe -m pytest          # hoặc scripts\run_tests.bat
```

53 test: hình học, 8 tình huống bắt buộc của association/hệ luật, bảng luật,
CSV/JSON/batch (kể cả ảnh hỏng), đánh giá, dò ngưỡng, và 3 test tích hợp với
`models\best.pt` (tự bỏ qua nếu thiếu mô hình).

Toàn bộ quy trình kiểm tra cuối (cài đặt → test → một ảnh → batch → đánh giá →
kiểm tra ảnh trùng lặp → gom hình vào `docs\figures\` → Streamlit) chạy bằng
**`scripts\run_all.bat`** (~4 phút trên CPU), log trong `logs\`.

## 7. Cấu trúc thư mục

```
helmet-violation-detection/
├── app/streamlit_app.py        giao diện demo
├── src/helmet_ppe/             mã nguồn (xem docs/02_system_architecture.md)
│   └── evaluation/             đánh giá, nhãn chuẩn, dò ngưỡng, ảnh lỗi
├── tests/                      pytest
├── scripts/                    *.bat cho Windows + check_streamlit.py
├── docs/                       tài liệu báo cáo (tiếng Việt)
├── models/                     best.pt (giữ nguyên), best_v2.pt
├── data/raw, data/test_images  ảnh thử nghiệm (không bị sửa; trùng ảnh Construction-PPE, xem docs/04)
├── data/construction-ppe/      dataset giải nén (tạo bởi prepare-dataset)
├── data/person_gt/             nhãn chuẩn cấp người
├── outputs/                    kết quả batch
├── reports/                    kết quả đánh giá
├── legacy/                     code thử nghiệm ban đầu (giữ lại để tham khảo)
├── requirements.txt, pyproject.toml
```

## 8. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: helmet_ppe` | Chưa đặt `PYTHONPATH=src` (hoặc dùng các tệp `.bat`). |
| `ModuleNotFoundError: ultralytics / streamlit` | Chạy `scripts\setup.bat` hoặc `pip install -r requirements.txt` trong `.venv`. |
| `Không tìm thấy mô hình: models\best.pt` | Đặt lại tệp trọng số vào `models\`. |
| Chữ tiếng Việt lỗi trong cmd | Các `.bat` đã `chcp 65001`; khi tự gõ lệnh, chạy `chcp 65001` trước. |
| CSV mở bằng Excel lỗi dấu | Tệp đã ghi UTF-8 có BOM; nếu vẫn lỗi, dùng Data → From Text/CSV, chọn UTF-8. |
| Không có người nào bị kết luận vi phạm | Kiểm tra `--conf-no-helmet` (lớp `no_helmet` cần ngưỡng thấp, mặc định 0,03). |
| Streamlit không mở trình duyệt | Mở thủ công <http://localhost:8501>; đổi cổng: `--server.port 8502`. |
| Chạy chậm | Mặc định chạy CPU (~45 ms suy luận/ảnh trên Core Ultra 7). Có GPU NVIDIA: cài torch bản CUDA, dùng `--device 0`. |

## 9. Tài liệu

| Tệp | Nội dung |
|---|---|
| [01_project_overview.md](docs/01_project_overview.md) | Đề tài, mục tiêu, phạm vi, công nghệ, đầu vào/đầu ra |
| [02_system_architecture.md](docs/02_system_architecture.md) | Kiến trúc, pipeline, module, luồng dữ liệu |
| [03_methodology.md](docs/03_methodology.md) | YOLO26n, IoU, thuật toán ghép, hệ luật |
| [04_dataset.md](docs/04_dataset.md) | Construction-PPE: số liệu thực, chia tập, hạn chế |
| [05_experiments.md](docs/05_experiments.md) | Môi trường, kịch bản, cách đo |
| [06_results.md](docs/06_results.md) | Bảng kết quả, biểu đồ, phân tích lỗi |
| [07_conclusion.md](docs/07_conclusion.md) | Kết quả đạt được, giới hạn, hướng cải tiến |
