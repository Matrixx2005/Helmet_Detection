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

Hệ thống dùng **một mô hình phát hiện chính** và **một mô hình hỗ trợ**:

| | `models/best.pt` (mô hình chính) | `models/yolo26n-pose.pt` (hỗ trợ) |
|---|---|---|
| Nhiệm vụ | Phát hiện `Person`, `helmet`, `no_helmet` | Phát hiện người + 17 điểm khớp (bổ sung người bị che, định vị đầu) |
| Nguồn | Fine-tune từ `yolo26n.pt` trên Google Colab (T4) | Ultralytics, huấn luyện sẵn trên COCO, không huấn luyện lại |
| Kiến trúc | YOLO26n | YOLO26n-pose |
| Tham số / GFLOPs | 2.538.746 (chưa fuse) · 5,4 GFLOPs | — |
| Kích thước ảnh vào | 640 | 640 |

### Siêu tham số huấn luyện của `best.pt`

Fine-tune (transfer learning) từ `yolo26n.pt` trên bộ huấn luyện (mục 4.1: 770 ảnh train, 230 ảnh val). Các giá trị đọc
từ `reports/training/yolo26n/args.yaml` và `results.csv`; notebook `notebooks/train_colab.ipynb`
(bản đã chạy và kết quả: `reports/training/yolo26n/colab_run_output.ipynb`).

| Tham số | Giá trị |
|---|---|
| Epochs cấu hình / thực tế | 100 (patience 20) / 75 (tốt nhất ở epoch 73) |
| Batch size · kích thước ảnh | 16 · 640 |
| Optimizer | AdamW (lr0 0,000667; lrf 0,01; momentum 0,937; weight decay 0,0005) |
| Warmup epochs / close mosaic | 3 / 10 |
| Trọng số loss (box / cls / dfl) | 7,5 / 0,5 / 1,5 |
| AMP · seed | Có · 0 |
| Phần cứng · thời gian | Tesla T4 · khoảng 22 phút (1.333 giây) |
| Phiên bản Ultralytics | 8.4.212 |

Tăng cường dữ liệu: mosaic 1,0 · lật ngang 0,5 · HSV (h 0,015; s 0,7; v 0,4) · translate 0,1 · scale 0,5 ·
random erasing 0,4 · RandAugment.

Kết quả ở epoch tốt nhất (tập val 230 ảnh của bộ huấn luyện): Precision 0,860 · Recall 0,773 · mAP@50 **0,819** ·
mAP@50-95 **0,597**.

### Hai điều chỉnh sau huấn luyện (ghi lại để minh bạch)

1. **Bộ dữ liệu đổi phiên bản.** Khi chạy notebook, tập Construction-PPE trên máy chủ Ultralytics đã là bản 11/2026
   (12 lớp, thứ tự khác bản cũ). Notebook ban đầu gán tên lớp theo thứ tự cũ nên checkpoint ghi sai tên (ví dụ lớp
   `Person` thực ra là `no_helmet`). Tên lớp được sửa đúng bằng `scripts/fix_model_names.py` (đồng thời bỏ phần optimizer,
   từ 26 MB xuống 5,4 MB); trọng số không đổi. Bản gốc từ Colab giữ ở `reports/training/yolo26n/best.pt`.
2. **Lớp thứ 12 (`no_vest`) không được học.** Mô hình có 11 lớp nên các ảnh có nhãn `no_vest` bị Ultralytics bỏ qua
   khi huấn luyện. Lớp này nằm ngoài phạm vi đề tài.

## 5.3. Kịch bản kiểm thử

| # | Kịch bản | Dữ liệu | Lệnh | Đầu ra |
|---|---|---|---|---|
| E1 | Kiểm thử đơn vị + tích hợp | dữ liệu giả + `best.pt` | `scripts\run_tests.bat` | `logs/pytest.log` |
| E2 | Đánh giá phát hiện YOLO | `data/ppe-eval` val (270 ảnh), test (255 ảnh) | `python -m helmet_ppe eval-detector` | `reports/detection_eval/` |
| E3 | Dò ngưỡng hệ luật | val, YOLO chạy ở conf 0,02 | `batch` + `tune-rules` | `reports/rule_tuning/<model>/` |
| E4 | Đánh giá suy luận cấp người | test, ngưỡng từ E3 | `batch --params` + `eval-rules` | `reports/rule_eval/<model>/` |
| E5 | Xử lý hàng loạt | `data/raw` (29 ảnh; trùng ảnh bộ đánh giá, xem 4.6) | `python -m helmet_ppe batch` | `outputs/experiment_01/` |
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
1. Chạy YOLO trên 270 ảnh val với ngưỡng 0,02 cho cả ba lớp, lưu mọi detection.
2. Thử **5.184 tổ hợp** (lưới `SweepGrid` trong `helmet_ppe/evaluation/tuning.py`): bật/tắt pose,
   `conf_pose_person` ∈ {0,3; 0,4; 0,5}, `conf_person` ∈ {0,3; 0,4; 0,5}, `conf_helmet` ∈ {0,2; 0,25; 0,3},
   `conf_no_helmet` ∈ {0,02; 0,03; 0,05; 0,1}, ngưỡng bằng chứng không-mũ ∈ {0; 0,02; 0,05} (bằng chứng mũ cố định 0,1,
   biên 0), dung sai đầu x ∈ {0,9; 1,2}, y ∈ {0,55; 0,8}, tỉ lệ kích thước đầu tối thiểu ∈ {0,5; 0,8}, bật/tắt bước bỏ
   hộp `no_helmet` dưới mũ.
3. Hàm mục tiêu: trung bình F1 của lớp "vi phạm" và lớp "đội mũ". Khi hòa, chọn
   ngưỡng cao hơn (thận trọng hơn).
4. Tham số tốt nhất ghi vào `best_params.json`, dùng **nguyên vẹn** cho tập test.

Lý do phải dò ngưỡng theo lớp: lớp `no_helmet` có confidence thấp một cách có hệ thống (mAP50 test 0,058, mục 6.1);
với ngưỡng chung 0,25 hệ thống bỏ sót gần một nửa số vi phạm (recall 0,532, mục 6.4).

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
hệ thống (`helmet_ppe/evaluation/baseline.py`); (c) hệ thống **không dùng pose** với
ngưỡng dò riêng trên val; (d) hệ thống đề xuất (có pose).
