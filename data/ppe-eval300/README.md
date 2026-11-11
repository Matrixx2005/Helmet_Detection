# Tập kiểm thử ppe-eval300

Tập val (300 ảnh) và test (300 ảnh) dùng để đánh giá mọi mô hình trong dự án. Định dạng YOLO, 11 lớp
(xem `data.yaml`); ảnh Roboflow chỉ có nhãn 3 lớp `helmet`, `Person`, `no_helmet`.

| Nguồn | Ảnh val | Ảnh test | Giấy phép |
|---|---|---|---|
| Construction-PPE (Ultralytics) | 143 | 141 | AGPL-3.0 |
| BAC_HIEN_CONSTRUCTION_SAFETY_2024 v8 (Roboflow Universe, `constructionsatefy`) | 157 | 159 | CC BY 4.0 |

`sources.csv` ghi nguồn từng ảnh. Ảnh Roboflow được chọn bằng `scripts/select_heldout.py`
(rõ nét, người đủ lớn, đa dạng, không trùng ảnh train); việc chọn ảnh rõ nét làm tập này dễ hơn
ảnh thực tế. Bộ này không được dùng để huấn luyện.

Nguồn Roboflow: https://universe.roboflow.com/constructionsatefy/bac_hien_construction_safety_2024/dataset/8
