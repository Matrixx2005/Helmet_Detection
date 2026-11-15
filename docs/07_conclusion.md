# 7. Kết luận

## 7.1. Kết quả đạt được

1. **Pipeline hoàn chỉnh** Ảnh → YOLO26n → ghép người–mũ → hệ luật → ảnh kết
   quả + CSV/JSON + biểu đồ, chạy được trên CPU (~30 ms suy luận YOLO/ảnh trên tập test).
2. **Thuật toán ghép** dựa trên vị trí tương đối, hiệu chỉnh từ nhãn chuẩn của
   Construction-PPE, ghép một-một toàn cục; xử lý đúng 8 tình huống bắt buộc
   (kiểm chứng bằng test tự động).
3. **Hệ luật thận trọng R0–R5**: chỉ kết luận nghi vi phạm khi có bằng chứng
   tích cực (`no_helmet`), mỗi kết luận kèm mã luật, lý do và số liệu bằng chứng.
4. **Phát hiện và xử lý một vấn đề thật của mô hình**: lớp `no_helmet` tổng quát hóa kém
   (mAP50 0,90 trên val của bộ huấn luyện nhưng chỉ 0,058 trên tập đánh giá độc lập). Dùng ngưỡng riêng theo lớp
   (dò trên val) kết hợp association và mô hình pose đưa F1 lớp vi phạm cấp người trên tập test lên **0,717**,
   cao hơn luật đối chứng ngây thơ (0,538) và cấu hình ngưỡng chung (0,646) — xem [06_results.md](06_results.md).
5. **Đánh giá trung thực**: tham số chọn trên val, báo cáo trên test; ảnh không
   có nhãn chỉ được thống kê, không gắn độ chính xác.
6. **Sản phẩm dùng được**: CLI, xử lý hàng loạt chịu lỗi từng ảnh, giao diện
   Streamlit tiếng Việt, 65 test tự động, tài liệu, script `.bat` một-nhấp.

## 7.2. Giới hạn của hệ thống

* **Phụ thuộc vào chất lượng phát hiện lớp `no_helmet`** (mAP50 test chỉ 0,058).
  Hệ thống phải dùng ngưỡng thấp (0,05) cho lớp này; độ tin cậy hiển thị trên các ca
  "nghi không đội mũ" vì vậy thấp. Có 11/79 người không đội mũ bị kết luận là đội mũ.
* **Nhãn chuẩn cấp người được suy ra tự động** bằng chính quy tắc ghép: con số
  đánh giá đo ảnh hưởng của lỗi phát hiện, chưa phải kiểm chứng độc lập của quy
  tắc ghép. Cần gán nhãn tay một tập nhỏ để có kết luận chắc chắn.
* **Tập test vừa phải** (79 người vi phạm) → khoảng tin cậy còn rộng; ±1 ca thay đổi
  precision/recall khoảng 1,3 điểm phần trăm. Kết quả đo trên bản Construction-PPE khác với bản dùng để huấn luyện
  nên phản ánh khả năng tổng quát hóa, không phải điểm tối đa của mô hình.
* **Bỏ sót người** (41/373 trên test) và dự đoán thừa (78); người bị bỏ sót thì không thể được kết luận.
* **Chưa có ảnh công trường tự thu thập**: 29 ảnh trong `data/raw` đều trùng ảnh của bộ đánh giá Construction-PPE
  (mô hình không học trên các ảnh này), nên chỉ có kết quả trên dữ liệu công khai, chưa có ảnh thực tế độc lập.
* **Miền dữ liệu**: mũ bảo hiểm xe đạp, ảnh đêm, ảnh xoay 90° gây lỗi; ngưỡng
  `conf_person = 0,4` làm giảm số người phát hiện trong ảnh đám đông xa camera.
* **Mô hình có 11 lớp nhưng bộ dữ liệu huấn luyện có 12**: lớp `no_vest` không được học (ngoài phạm vi đề tài).
* Chỉ xử lý **ảnh tĩnh**, không theo dõi người qua các khung hình video.

## 7.3. Hướng cải tiến

1. **Cải thiện lớp `no_helmet`**: huấn luyện lại với trọng số lớp / oversampling
   ảnh có `no_helmet`, bổ sung dữ liệu từ nhiều nguồn (ví dụ SHWD, Hard Hat Workers, bộ BAC_HIEN trên Roboflow Universe), hoặc dùng
   mô hình lớn hơn (YOLO26s/m) — đây là nút thắt lớn nhất.
2. **Bộ phân loại vùng đầu**: cắt vùng đầu ước lượng của mỗi người và phân loại
   có/không mũ bằng một CNN nhỏ, thay vì chỉ dựa vào hộp `no_helmet`.
3. **Thu thập ảnh công trường mới** (không thuộc Construction-PPE) và **gán nhãn
   tay cấp người** (định dạng CSV đã hỗ trợ) để đánh giá độc lập.
4. **Video**: thêm tracking (ByteTrack có sẵn trong Ultralytics) và luật theo thời
   gian (vi phạm kéo dài N khung hình mới cảnh báo) để giảm báo nhầm.
5. Hiệu chỉnh ngưỡng theo từng camera/công trường; xuất ONNX/OpenVINO để tăng tốc
   trên CPU Intel.
