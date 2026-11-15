# 4. Bộ dữ liệu Construction-PPE

Đề tài dùng **hai phiên bản** của bộ Construction-PPE (Ultralytics, giấy phép AGPL-3.0, tài liệu
<https://docs.ultralytics.com/datasets/detect/construction-ppe/>): một phiên bản để **huấn luyện** và một
phiên bản khác để **đánh giá**. Hai phiên bản không có ảnh nào trùng nhau.

## 4.1. Nguồn và vai trò

| | Bộ huấn luyện | Bộ đánh giá |
|---|---|---|
| Phiên bản | Ultralytics, phát hành 11/2026 (tải tự động bằng `notebooks/train_colab.ipynb`) | Ultralytics, bản `construction-ppe.zip` (178,4 MB) tải trước đó |
| Số ảnh | 1.000 (707 ảnh `ultralytics-ppe-nov2026-*` + 293 ảnh `source-*`; 770 ảnh dùng huấn luyện) | 1.416 (chia lại, xem 4.3) |
| Số lớp | 12 | 11 |
| Dùng để | Huấn luyện YOLO26n | Chọn ngưỡng (val), báo cáo kết quả (test) |

Kiểm tra bằng perceptual hash 16×16 (khoảng cách Hamming ≤ 5/256): **0/891 ảnh train, 0/270 ảnh val và
0/255 ảnh test của bộ đánh giá có bản trùng hoặc gần trùng trong 1.000 ảnh của bộ huấn luyện.** Vì vậy
tập val và test không bị rò rỉ từ quá trình huấn luyện.

> **Lưu ý thứ tự lớp.** Hai phiên bản đánh số lớp khác nhau (bản 11/2026 có `forklift`, `no_vest` và bỏ
> `none`; `Person` là lớp 5 thay vì 6). Mô hình `models/best.pt` dùng thứ tự của **bản mới**; nhãn của bộ đánh giá
> được chuyển về cùng thứ tự bằng `scripts/convert_labels_new_scheme.py` (ghi sang `data/ppe-eval/`, bộ gốc
> không bị sửa). Tên lớp trong checkpoint được sửa bằng `scripts/fix_model_names.py` vì notebook huấn luyện ban
> đầu gán tên theo thứ tự cũ (xem 5.2).

Thứ tự lớp dùng trong toàn bộ tài liệu này: 0 `helmet`, 1 gloves, 2 vest, 3 boots, 4 goggles, **5 `Person`**,
**6 `no_helmet`**, 7 no_goggle, 8 no_gloves, 9 no_boots, 10 forklift (và 11 no_vest ở bộ huấn luyện).
Hệ thống chỉ dùng ba lớp in đậm/mã hóa: `helmet`, `Person`, `no_helmet`.

## 4.2. Cấu trúc

```
construction-ppe/
├── data.yaml          đường dẫn split + tên lớp
├── images/{train,val,test}/*.jpg|*.jpeg
└── labels/{train,val,test}/*.txt   định dạng YOLO: class cx cy w h (chuẩn hóa 0..1)
```

## 4.3. Chia tập

### Bộ đánh giá (bản `construction-ppe.zip`)

Bộ gốc chia sẵn train 1.132 / val 143 / test 141 ảnh. Để có tập val và test đủ lớn, đề tài **chia lại** (seed
cố định, `scripts/resplit_construction_ppe.py`, lịch sử ghi trong `reports/construction_ppe_resplit*.csv`):

| Bước | Train | Val | Test |
|---|---|---|---|
| Gốc (Ultralytics) | 1.132 | 143 | 141 |
| Gộp train + test, chia lại 4 : 1 | 1.018 | 143 | 255 |
| Chuyển 127 ảnh train sang val | **891** | **270** | **255** |

Chỉ **val (270 ảnh)** và **test (255 ảnh)** được dùng để đánh giá (`data/ppe-eval/`, nhãn đã đổi sang thứ tự lớp
mới). Tập train 891 ảnh của bộ này **không** dùng để huấn luyện mô hình cuối cùng.

### Bộ huấn luyện (bản 11/2026)

Notebook chia 77% / 23% theo tên ảnh đã sắp xếp: **770 ảnh train, 230 ảnh val** (val dùng để chọn checkpoint).
Val của bộ này chỉ gồm ảnh `ultralytics-ppe-nov2026-*`; train gồm toàn bộ 293 ảnh `source-*` và 477 ảnh
`ultralytics-ppe-nov2026-*`.

## 4.4. Các lớp và số lượng

Định dạng ô: *số hộp (số ảnh chứa lớp đó)*. Chỉ liệt kê đầy đủ ba lớp hệ thống sử dụng.

| Lớp | Huấn luyện: train (770 ảnh) | Huấn luyện: val (230 ảnh) | Đánh giá: val (270 ảnh) | Đánh giá: test (255 ảnh) |
|---|---|---|---|---|
| **helmet** | 1.357 | 219 | 344 (193) | 284 (187) |
| **Person** | 2.116 | 327 | 432 (258) | 373 (240) |
| **no_helmet** | 725 | 88 | 96 (61) | 82 (52) |
| gloves | 1.523 | 406 | 250 | 272 |
| vest | 1.584 | 232 | 303 | 277 |
| boots | 2.084 | 400 | 272 | 304 |
| goggles | 537 | 214 | 86 | 103 |
| no_goggle | 261 | 86 | 89 | 79 |
| no_gloves | 867 | 175 | 124 | 101 |
| no_boots | 570 | 177 | 11 | 12 |
| forklift | 278 | 70 | — | — |
| no_vest | 485 | 82 | — | — |

(Mô hình được huấn luyện với 11 lớp nên nhãn `no_vest` — lớp 11 — không được học; 71 ảnh val của bộ
huấn luyện có nhãn này bị Ultralytics bỏ qua khi đánh giá.)

## 4.5. Nhãn chuẩn cấp người

Bộ dữ liệu không có nhãn "người này có đội mũ không", chỉ có nhãn hộp. Hệ thống suy ra nhãn cấp người
(`data/person_gt/*.csv`) bằng cách áp dụng **cùng quy tắc ghép hình học** lên các hộp chuẩn (confidence = 1):

| Split | Người | HELMET | NO_HELMET | UNDETERMINED |
|---|---|---|---|---|
| val (270 ảnh) | 432 | 280 | 96 | 56 |
| test (255 ảnh) | 373 | 241 | 79 | 53 |

`UNDETERMINED` = người không có hộp `helmet` / `no_helmet` chuẩn nào ở vùng đầu (quay lưng, đầu bị che, hoặc
người gán nhãn bỏ qua); các người này không dùng để tính Precision/Recall.

Vì nhãn cấp người được suy ra bằng chính quy tắc ghép của hệ thống, kết quả đánh giá đo **ảnh hưởng của lỗi phát
hiện YOLO** lên kết luận cuối, chưa phải kiểm chứng độc lập của quy tắc ghép.

## 4.6. Hạn chế của bộ dữ liệu

1. **Lệch miền dữ liệu giữa huấn luyện và đánh giá.** Mô hình học trên bản 11/2026 nhưng được đo trên bản cũ
   (cảnh, góc chụp và cách gán nhãn khác). Điều này khiến điểm trên test thấp hơn điểm trên val của chính bộ
   huấn luyện (mục 6.1) — đó là phép đo khả năng tổng quát hóa, không phải lỗi mô hình.
2. **Lớp `no_helmet` hiếm**: 725 hộp train so với 1.357 `helmet`; chỉ 82 hộp ở test, nên chỉ số của lớp vi phạm có
   phương sai lớn.
3. **Gán nhãn `Person` không nhất quán**: có ảnh khung người bắt đầu từ vai (không chứa đầu), ảnh đám đông chỉ
   gán một phần số người, và khung người chuẩn đôi khi bao nhiều người cùng lúc (xem 6.5).
4. **Ảnh xoay 90°** và ảnh ghép xuất hiện trong dữ liệu, làm giả định "đầu ở phía trên khung người" không đúng cho
   các ảnh này.
5. Ảnh chủ yếu là công trường nước ngoài, góc chụp gần.
6. **Ảnh trong `data/raw` không độc lập với bộ đánh giá** (`data/test_images` là 26 ảnh trong số đó):
   `scripts/find_dataset_duplicates.py` cho thấy cả 29 ảnh đều trùng ảnh của bộ đánh giá — 18 ảnh train, 8 val, 3 test
   (`reports/data_raw_duplicates.csv`). Do mô hình không học trên bộ đánh giá nên 29 ảnh này không bị rò rỉ vào huấn
   luyện, nhưng kết quả trên chúng chỉ mang tính minh họa.
7. **Gần trùng giữa các split của bộ đánh giá**: 10/270 ảnh val và 18/255 ảnh test có bản gần trùng trong train gốc
   (`reports/dataset_split_overlap.csv`). Vì mô hình không dùng train này để học nên không ảnh hưởng tới tính hợp lệ
   của test; chỉ cần lưu ý khi so sánh với các phép chia khác.
