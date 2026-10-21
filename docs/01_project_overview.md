# 1. Tổng quan đề tài

## 1.1. Tên đề tài

**Phát hiện mũ bảo hộ lao động và suy luận vi phạm bằng hệ luật.**

## 1.2. Bối cảnh

Trên công trường, đội mũ bảo hộ là quy định an toàn bắt buộc. Giám sát thủ công
tốn công và dễ bỏ sót. Đề tài xây dựng một hệ thống nhận ảnh công trường, phát
hiện người và mũ bảo hộ bằng mạng học sâu YOLO, rồi dùng một **hệ luật** để suy
luận từng người có đang đội mũ hay không.

Điểm khác biệt so với việc chỉ chạy YOLO: YOLO chỉ trả về các hộp rời rạc
("có một người ở đây", "có một mũ ở kia"). Hệ thống phải trả lời câu hỏi
nghiệp vụ *"người số 2 có đội mũ không?"*, nên cần thêm hai bước:

1. **Ghép (association)** mũ với đúng người dựa trên hình học bounding box.
2. **Suy luận (rule engine)** trạng thái của từng người, thận trọng khi bằng
   chứng chưa đủ.

## 1.3. Mục tiêu

| # | Mục tiêu | Kết quả tương ứng |
|---|---|---|
| 1 | Phát hiện người, mũ, vùng đầu không mũ bằng YOLO26n | `src/helmet_ppe/detector.py` |
| 2 | Ghép mũ với đúng người | `src/helmet_ppe/association.py` |
| 3 | Suy luận trạng thái đội mũ | `src/helmet_ppe/rule_engine.py` |
| 4 | Hiển thị bounding box, Person ID, trạng thái, độ tin cậy | `src/helmet_ppe/visualize.py` |
| 5 | Xử lý hàng loạt, xuất CSV/JSON/biểu đồ | `src/helmet_ppe/batch.py` |
| 6 | Đánh giá định lượng khi có nhãn | `src/helmet_ppe/evaluation/` |
| 7 | Giao diện demo | `app/streamlit_app.py` |

## 1.4. Phạm vi

**Trong phạm vi:** ảnh tĩnh; ba lớp `Person`, `helmet`, `no_helmet`; suy luận
trạng thái đội mũ cho từng người.

**Ngoài phạm vi:** áo phản quang, găng tay, giày, kính (mô hình có các lớp này
nhưng hệ thống bỏ qua); video/webcam thời gian thực; theo dõi (tracking); huấn
luyện lại mô hình.

## 1.5. Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.13 |
| Mô hình phát hiện | Ultralytics YOLO26 Nano (`yolo26n`), huấn luyện trên Construction-PPE |
| Xử lý ảnh | OpenCV, Pillow, NumPy |
| Thống kê, biểu đồ | pandas, matplotlib |
| Giao diện | Streamlit |
| Kiểm thử | pytest |
| Môi trường | Windows 11, VS Code, virtualenv `.venv` |

## 1.6. Đầu vào và đầu ra

**Đầu vào:** một ảnh hoặc một thư mục ảnh (JPG, JPEG, PNG, WEBP, BMP).

**Đầu ra** (cho mỗi lần chạy, trong `outputs/<tên_thí_nghiệm>/`):

| Tệp | Nội dung |
|---|---|
| `annotated_images/` | Ảnh đã vẽ khung người (màu theo trạng thái), mũ, nhãn `P1 HELMET 0.87` |
| `detections.csv` | Từng đối tượng YOLO phát hiện, kèm vai trò sau khi ghép |
| `persons.csv` | Từng người: trạng thái, luật áp dụng, bằng chứng, lý do |
| `summary.csv` | Thống kê từng ảnh |
| `results.json` | Toàn bộ thông tin dạng JSON + thống kê tổng |
| `errors.csv` | Ảnh lỗi và nguyên nhân |
| `config.json` | Thông số thí nghiệm (mô hình, ngưỡng, môi trường) |
| `charts/` | Biểu đồ PNG cho báo cáo |

Bốn trạng thái kết luận:

| Trạng thái | Ý nghĩa |
|---|---|
| `HELMET_DETECTED` | Có bằng chứng mũ được đội trên đầu |
| `SUSPECTED_NO_HELMET` | Có bằng chứng tích cực là vùng đầu không đội mũ (nghi vi phạm) |
| `UNKNOWN` | Không đủ bằng chứng để kết luận |
| `NO_PERSON_DETECTED` | Ảnh không có người |
