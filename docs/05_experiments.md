# 5. Thực nghiệm

## 5.1. Môi trường thực nghiệm

Mọi số liệu trong báo cáo được đo trên máy của sinh viên:

| Thành phần | Giá trị |
|---|---|
| CPU | Intel Core Ultra 7 155H (không dùng GPU) |
| Hệ điều hành | Windows 11 |
| Python | 3.13.11 (virtualenv `.venv`) |
| PyTorch | 2.14.1+cpu |
| Ultralytics | 8.4.177 |
| Streamlit | 1.66.0 · pytest 9.1.1 |

## 5.2. Mô hình

| | `models/best.pt` (mô hình chính) | `models/best_v2.pt` (đối chứng) |
|---|---|---|
| Nguồn | Mô hình gốc của đề tài | `Downloads\best.pt`, lần huấn luyện `helmet_yolo26n_v2` |
| Kiến trúc | YOLO26n | YOLO26n |
| Tham số / GFLOPs | 2.392.165 / 5,4 | 2.392.165 / 5,4 |
| Kích thước ảnh vào | 640 | 640 |

`models/best.pt` được giữ nguyên (không ghi đè, không huấn luyện lại).
`best_v2.pt` là bản sao, chỉ dùng để so sánh.

## 5.3. Kịch bản kiểm thử

| # | Kịch bản | Dữ liệu | Lệnh | Đầu ra |
|---|---|---|---|---|
| E1 | Kiểm thử đơn vị + tích hợp | dữ liệu giả + `best.pt` | `scripts\run_tests.bat` | `logs/pytest.log` |
| E2 | Đánh giá phát hiện YOLO | Construction-PPE val (143 ảnh), test (141 ảnh) | `python -m helmet_ppe eval-detector` | `reports/detection_eval/` |
| E3 | Dò ngưỡng hệ luật | val, YOLO chạy ở conf 0,02 | `batch` + `tune-rules` | `reports/rule_tuning/<model>/` |
| E4 | Đánh giá suy luận cấp người | test, ngưỡng từ E3 | `batch --params` + `eval-rules` | `reports/rule_eval/<model>/` |
| E5 | Xử lý hàng loạt | `data/raw` (29 ảnh; trùng ảnh Construction-PPE, xem 4.6) | `python -m helmet_ppe batch` | `outputs/experiment_01/` |
| E6 | Giao diện | — | `scripts/check_streamlit.py` | `logs/streamlit_check.log` |
| E7 | So sánh cấu hình (ablation) | test | `eval-rules --baseline`, `compare-rules` | `reports/rule_eval/comparison/` |

Toàn bộ chuỗi E1–E7 chạy bằng một lệnh: `scripts\run_all.bat` (khoảng 4 phút trên CPU).

### Tám tình huống bắt buộc của Association / Rule Engine (E1)

Mỗi tình huống là một test trong `tests/test_rule_engine.py` và
`tests/test_association.py`, dựng cảnh bằng hộp tọa độ giả (không cần mô hình):

| Tình huống | Kết quả mong đợi | Test |
|---|---|---|
| Một người đội mũ | `HELMET_DETECTED` (R1) | `test_one_person_wearing_helmet` |
| Một người không thấy mũ | `UNKNOWN` (R5), **không** phải vi phạm | `test_one_person_without_visible_helmet_is_unknown_not_violation` |
| Hai người đứng cạnh nhau | Mỗi người nhận đúng mũ của mình | `test_two_people_side_by_side_*` |
| Một người đội mũ, một người không | `HELMET_DETECTED` + `SUSPECTED_NO_HELMET` | `test_one_with_helmet_one_without` |
| Người cầm mũ trên tay | Mũ không được ghép; `UNKNOWN` | `test_person_holding_helmet_in_hand_*`, `test_helmet_held_in_hand_is_not_matched` |
| Mũ không thuộc người nào | Mũ nằm trong `unmatched_helmet_ids` | `test_helmet_not_belonging_to_anyone`, `test_helmet_far_from_everyone_is_unmatched` |
| Người bị che khuất / cắt mép | `UNKNOWN` + cờ `touches_top_edge` | `test_occluded_person_cut_at_top_edge_is_unknown_with_flag` |
| Không có người | `NO_PERSON_DETECTED` (R0) | `test_image_without_person` |

Thêm: hai người chồng lấn (mũ về đúng người phía sau), một mũ không bao giờ ghép
cho hai người, bảng luật R1–R5 (8 trường hợp), mâu thuẫn mũ/không-mũ (R3),
mũ độ tin cậy thấp (R4).

## 5.4. Cách thu thập kết quả

### Đánh giá phát hiện (E2)
`YOLO.val()` của Ultralytics trên split val/test với `conf = 0,001`, `iou = 0,7`,
`imgsz = 640`. Thu: Precision, Recall, mAP50, mAP50-95 tổng (trung bình 11 lớp)
và từng lớp, confusion matrix, đường PR/F1. Không có số liệu nào được nhập tay —
tất cả đọc từ đối tượng kết quả của Ultralytics và ghi ra CSV/JSON.

### Nhãn chuẩn cấp người
Suy ra từ nhãn hộp chuẩn (mục 4.5). Người dự đoán được ghép với người chuẩn bằng
IoU ≥ 0,5 (tham lam một-một).

### Dò ngưỡng (E3) — chỉ trên tập val
1. Chạy YOLO trên 143 ảnh val với ngưỡng 0,02 cho cả ba lớp, lưu mọi detection.
2. Thử **1.792 tổ hợp**: `conf_person` ∈ {0,25; 0,3; 0,4; 0,5},
   `conf_helmet` ∈ {0,2; 0,25; 0,3; 0,4},
   `conf_no_helmet` ∈ {0,02; 0,03; 0,05; 0,08; 0,1; 0,15; 0,25},
   ngưỡng bằng chứng mũ ∈ {0,1; 0,2}, ngưỡng bằng chứng không-mũ ∈ {0; 0,02; 0,03; 0,05},
   biên ∈ {0; 0,1}.
3. Hàm mục tiêu: trung bình F1 của lớp "vi phạm" và lớp "đội mũ". Khi hòa, chọn
   ngưỡng cao hơn (thận trọng hơn).
4. Tham số tốt nhất ghi vào `best_params.json`, dùng **nguyên vẹn** cho tập test.

Lý do phải dò ngưỡng theo lớp: ở ngưỡng chung 0,25, mô hình **không trả về hộp
`no_helmet` nào** trên tập test (recall lớp này = 0, mục 6.1) nên hệ luật không
bao giờ có bằng chứng vi phạm. Lớp này hiếm (400 hộp train so với 1.357 `helmet`)
nên confidence của mô hình thấp một cách có hệ thống.

### Chỉ số cấp người (E4)
* **Precision / Recall / F1** cho lớp *vi phạm* (`NO_HELMET` ↔ `SUSPECTED_NO_HELMET`)
  và lớp *đội mũ* (`HELMET` ↔ `HELMET_DETECTED`). Recall tính trên toàn bộ người
  chuẩn của lớp, nên người bị bỏ sót hoặc kết luận `UNKNOWN` đều bị tính là sai.
  Người dự đoán không khớp ai (*false person*) mà bị kết luận thì tính vào FP.
* **Số người bỏ sót**, **tỉ lệ UNKNOWN** (trên người khớp được), **độ chính xác
  khi đã quyết định** (bỏ qua UNKNOWN).
* Ma trận nhầm lẫn và danh sách ca sai (`error_cases.csv`), ảnh minh họa ca sai
  (`error_examples/`).

### Ảnh trong `data/raw` (E5)
Chỉ thống kê (số người, số mũ, phân bố trạng thái, thời gian). **Không** tính độ
chính xác: thư mục không có nhãn đi kèm và các ảnh trùng với Construction-PPE.

### So sánh cấu hình (E7)
Trên cùng tập test và cùng nhãn chuẩn, so sánh: (a) `best.pt` với ngưỡng chung
0,25 và luật mặc định ban đầu (chưa dò); (b) luật đối chứng ngây thơ "tâm mũ nằm
trong khung người → đội mũ, ngược lại → vi phạm" chạy trên chính các detection của
hệ thống (`helmet_ppe/evaluation/baseline.py`); (c) `best_v2.pt` với ngưỡng dò
riêng trên val; (d) hệ thống đề xuất.
