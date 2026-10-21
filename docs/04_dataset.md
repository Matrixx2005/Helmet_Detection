# 4. Bộ dữ liệu Construction-PPE

## 4.1. Nguồn

**Construction-PPE** do Ultralytics phát hành (giấy phép AGPL-3.0), tài liệu:
<https://docs.ultralytics.com/datasets/detect/construction-ppe/>. Tệp sử dụng:
`construction-ppe.zip` (178,4 MB) tải từ trang Ultralytics. Mô hình
`models/best.pt` được huấn luyện trên bộ dữ liệu này (đường dẫn
`construction-ppe.yaml` ghi trong checkpoint).

Mọi số liệu bên dưới được **đếm trực tiếp** từ tệp zip (script
`python -m helmet_ppe prepare-dataset` giải nén vào `data/construction-ppe/`,
không sửa tệp gốc).

## 4.2. Cấu trúc

```
construction-ppe/
├── data.yaml          đường dẫn split + tên 11 lớp
├── images/{train,val,test}/*.jpg|*.jpeg
└── labels/{train,val,test}/*.txt   định dạng YOLO: class cx cy w h (chuẩn hóa 0..1)
```

## 4.3. Chia tập (train / val / test)

Bộ dữ liệu được chia sẵn bởi Ultralytics; đề tài giữ nguyên.

| Split | Số ảnh | Số tệp nhãn | Vai trò trong đề tài |
|---|---|---|---|
| train | 1.132 | 1.142 (*) | Huấn luyện mô hình (đã làm trước) + hiệu chỉnh hình học |
| val | 143 | 143 | Đánh giá mô hình, **dò ngưỡng hệ luật** |
| test | 141 | 141 | **Báo cáo kết quả cuối cùng** (không dùng để chọn tham số) |

(*) 10 tệp nhãn train dạng `image95(1).txt` không có ảnh tương ứng — được YOLO bỏ qua.
Ảnh có đuôi `.jpg` (1.013) và `.jpeg` (119) ở tập train.

## 4.4. Các lớp và số lượng

Định dạng ô: *số hộp (số ảnh chứa lớp đó)*. Ba lớp in đậm là lớp hệ thống sử dụng.

| ID | Lớp | train | val | test |
|---|---|---|---|---|
| **0** | **helmet** | 1.357 (840) | 201 (107) | 192 (110) |
| 1 | gloves | 1.162 (571) | 136 (68) | 163 (75) |
| 2 | vest | 1.283 (837) | 171 (109) | 178 (111) |
| 3 | boots | 1.251 (538) | 151 (64) | 211 (75) |
| 4 | goggles | 427 (393) | 47 (44) | 52 (51) |
| 5 | none | 654 (349) | 81 (43) | 65 (35) |
| **6** | **Person** | 1.790 (1.077) | 239 (139) | 236 (133) |
| **7** | **no_helmet** | 400 (232) | 45 (27) | 40 (24) |
| 8 | no_goggle | 337 (216) | 41 (25) | 33 (24) |
| 9 | no_gloves | 442 (190) | 56 (23) | 58 (23) |
| 10 | no_boots | 88 (28) | 4 (2) | 23 (6) |

## 4.5. Nhãn chuẩn cấp người

Bộ dữ liệu không có nhãn "người này có đội mũ không", chỉ có nhãn hộp. Hệ thống
suy ra nhãn cấp người (`data/person_gt/*.csv`) bằng cách áp dụng **cùng quy tắc
ghép hình học** lên các hộp chuẩn (confidence = 1):

| Split | Người | HELMET | NO_HELMET | UNDETERMINED |
|---|---|---|---|---|
| val | 239 | 163 | 45 | 31 |
| test | 236 | 156 | 40 | 40 |

`UNDETERMINED` = người không có hộp `helmet` / `no_helmet` chuẩn nào ở vùng
đầu (ví dụ quay lưng, đầu bị che, hoặc người gán nhãn bỏ qua) — các người này
không được dùng để tính Precision/Recall.

Kiểm tra trên nhãn chuẩn tập test: 156/192 hộp mũ được ghép với một người; 36 mũ
không ghép được vì ảnh không có hộp `Person` (17 trường hợp, tính cả val), mũ
nằm ngoài mọi khung người (15), nhiều mũ hơn số người được gán nhãn trong đám
đông, hoặc mũ nằm thấp (cầm tay). 40/40 hộp `no_helmet` test đều ghép được.

## 4.6. Hạn chế của bộ dữ liệu

1. **Lớp `no_helmet` hiếm**: chỉ 400 hộp ở train (so với 1.357 `helmet`), 40 hộp ở
   test → chỉ số của lớp vi phạm có phương sai lớn.
2. **Gán nhãn `Person` không nhất quán**: có ảnh khung người bắt đầu từ vai
   (không chứa đầu), có ảnh đám đông chỉ gán nhãn một phần số người. Khoảng
   2,7% mũ (41/≈1.550 ở train+val) nằm hoàn toàn phía trên khung người. Nới
   điều kiện để bắt các ca này hầu như không tăng số ghép đúng (thử nghiệm
   `min_rel_y` = −0,15 / −0,25 / −0,35 cho 1.285 / 1.286 / 1.286 cặp), nên giữ −0,15.
3. **Ảnh xoay 90°** và ảnh ghép xuất hiện trong dữ liệu (tăng cường dữ liệu),
   làm giả định "đầu ở phía trên khung người" không đúng cho các ảnh này.
4. Ảnh chủ yếu là công trường nước ngoài, góc chụp gần.
5. Tệp `annotations.zip` trong thư mục Downloads (5.000 tệp XML Pascal VOC của
   bộ *Hard Hat Workers*, không kèm ảnh) **không** được dùng.
6. **Ảnh trong `data/raw` không độc lập với bộ dữ liệu** (`data/test_images` là
   26 ảnh trong số đó): `scripts/find_dataset_duplicates.py` (perceptual hash
   16×16) cho thấy cả 29 ảnh đều trùng ảnh của Construction-PPE — 20 test, 4 val,
   5 train (`reports/data_raw_duplicates.csv`). Kết quả trên các ảnh này chỉ mang
   tính minh họa.
7. **Rò rỉ nhẹ giữa các split**: cùng phương pháp cho thấy 8/141 ảnh test và
   7/143 ảnh val có bản gần trùng trong train (`reports/dataset_split_overlap.csv`),
   nên số liệu trên test có thể lạc quan đôi chút.
