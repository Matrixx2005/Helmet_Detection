# 3. Phương pháp

## 3.1. YOLO26n

YOLO (*You Only Look Once*) là họ mô hình phát hiện đối tượng một giai đoạn: một
lần lan truyền qua mạng cho ra đồng thời vị trí (bounding box), lớp và độ tin
cậy của mọi đối tượng trong ảnh. Hệ thống dùng **YOLO26 Nano** của Ultralytics
(tệp cấu hình `yolo26n.yaml`), phiên bản nhỏ nhất của thế hệ YOLO26:

| Thông số (đo được khi nạp `models/best.pt`) | Giá trị |
|---|---|
| Số lớp | 121 lớp mạng (sau khi fuse) |
| Số tham số | 2.392.165 |
| Khối lượng tính toán | 5,4 GFLOPs (ảnh 640×640) |
| Số lớp đối tượng | 11 (Construction-PPE) |

Mô hình được huấn luyện sẵn (fine-tune từ `yolo26n.pt`) trên Construction-PPE
— đề tài **không** huấn luyện lại. Hệ thống chỉ dùng 3/11 lớp: `Person` (6),
`helmet` (0) và `no_helmet` (7).

## 3.2. Bounding box, Confidence và IoU

* **Bounding box** được biểu diễn `(x1, y1, x2, y2)` theo pixel: góc trên-trái
  và góc dưới-phải. Trục y hướng xuống.
* **Confidence** ∈ [0, 1]: độ tin cậy mô hình gán cho một hộp. Hộp có
  confidence nhỏ hơn ngưỡng `conf` bị loại.
* **IoU** (*Intersection over Union*) đo mức chồng lấn của hai hộp:

  ```
  IoU(A, B) = diện tích(A ∩ B) / diện tích(A ∪ B)
  ```

  IoU được dùng ở ba chỗ: (1) NMS bên trong YOLO loại các hộp trùng (ngưỡng
  `iou = 0.7`); (2) khi đánh giá mAP, một dự đoán là đúng nếu IoU với hộp chuẩn
  ≥ 0,5 (mAP50) hoặc trung bình trên các ngưỡng 0,5…0,95 (mAP50-95); (3) khi đánh
  giá hệ luật, người dự đoán được ghép với người chuẩn nếu IoU ≥ 0,5.

* **Tỉ lệ nằm trong** (`inside_ratio`): phần diện tích của hộp A nằm trong hộp B,
  chia cho diện tích A. Khác IoU, chỉ số này không bị "phạt" khi B lớn hơn A
  nhiều — phù hợp để hỏi "mũ có nằm trong khung người không?".

## 3.3. Thuật toán Association (ghép mũ với người)

### Vì sao không chỉ kiểm tra "mũ nằm trong khung người"?

Cách đơn giản đó sai trong nhiều tình huống: hai người đứng sát nhau khiến một
mũ nằm trong cả hai khung; người cầm mũ trên tay có mũ nằm trong khung nhưng
không ở trên đầu; mũ của người phía sau lọt vào khung người phía trước.

### Tọa độ tương đối

Với người có khung `(px1, py1, px2, py2)`, rộng `w`, cao `h`, và vật thể (mũ
hoặc `no_helmet`) có tâm `(cx, cy)`:

```
rel_y       = (cy − py1) / h          # 0 = đỉnh khung người, 1 = chân
rel_x       = (cx − tâm_x_người) / (w/2)   # 0 = giữa, ±1 = mép trái/phải
width_ratio = bề rộng vật thể / w
```

### Hiệu chỉnh từ dữ liệu

Các ngưỡng không chọn tùy ý mà được đo trên **nhãn chuẩn** của tập train + val
Construction-PPE (1.558 mũ, 445 vùng no_helmet):

| Đại lượng (mũ nằm trong khung người) | P5 | Trung vị | P95 |
|---|---|---|---|
| `rel_y` của tâm mũ | 0,045 | **0,108** | 0,266 |
| `rel_x` | −0,60 | −0,03 | 0,60 |
| `width_ratio` | 0,22 | 0,42 | 0,73 |

→ đầu nằm quanh 11% chiều cao khung người tính từ đỉnh; 95% mũ có tâm cao
hơn 27% chiều cao. (Khung người trong dataset thường chỉ là nửa thân trên, nên
tỉ lệ này lớn hơn tỉ lệ đầu/thân ở người đứng thẳng ~1/8.)

### Bước 1 – Loại cặp không hợp lệ (gate)

Một cặp (người, vật thể) bị loại nếu vi phạm bất kỳ điều kiện nào:

| Điều kiện | Ngưỡng | Ý nghĩa |
|---|---|---|
| `rel_y` ∈ [−0,15; 0,40] | `min_rel_y`, `max_rel_y` | Tâm vật thể phải ở vùng đầu (cho phép nhô lên trên khung 15%). Mũ **cầm trên tay** ở ngang ngực/thắt lưng có `rel_y` > 0,4 nên bị loại. |
| \|`rel_x`\| ≤ 1,05 | `max_abs_rel_x` | Tâm nằm trong bề ngang khung người |
| `width_ratio` ∈ [0,06; 1,30] | | Kích thước hợp lý so với người |
| `inside_ratio` ≥ 0,5 | | Ít nhất nửa vật thể nằm trong khung người (đã nới lên trên) |

### Bước 2 – Chấm điểm ghép

```
s_dọc   = max(0, 1 − |rel_y − 0,11| / 0,25)
s_ngang = max(0, 1 − min(|rel_x|, 1))
điểm    = 0,5·s_dọc + 0,3·s_ngang + 0,2·inside_ratio      ∈ [0, 1]
```

Vị trí dọc được ưu tiên nhất vì đó là đặc trưng phân biệt "đội trên đầu" với
"cầm trên tay". Cặp có điểm < 0,35 không được ghép.

### Bước 3 – Ghép một-một tham lam

Mọi cặp hợp lệ được sắp theo điểm giảm dần; lần lượt chọn cặp nếu cả người và
vật thể đều chưa được dùng. Kết quả:

* Mỗi mũ thuộc tối đa **một** người → không ghép một mũ cho nhiều người.
* Khi hai người đứng sát nhau, mũ được trao cho người có vùng đầu khớp nhất
  (điểm cao nhất), không phụ thuộc thứ tự duyệt.
* Mũ không thuộc ai được trả về riêng (`unmatched`) — ví dụ mũ để dưới đất, mũ
  cầm tay, hoặc người đội mũ bị YOLO bỏ sót.

Mũ và `no_helmet` được ghép **độc lập** với nhau; nếu một người có cả hai, hệ
luật sẽ phân xử.

### Lưu bằng chứng

Mỗi ghép lưu: `det_id`, confidence YOLO, điểm ghép và **độ mạnh bằng chứng**
`= confidence × điểm ghép`. Tất cả xuất ra `persons.csv` và `results.json`.

## 3.4. Hệ luật suy luận

Ký hiệu `H` = độ mạnh bằng chứng mũ, `N` = độ mạnh bằng chứng no_helmet của một
người (0 nếu không có). Luật áp dụng theo thứ tự, luật đầu tiên khớp quyết định:

| Luật | Điều kiện | Kết luận |
|---|---|---|
| R0 | Ảnh không có người | `NO_PERSON_DETECTED` |
| R1 | `H ≥ τ` và `H ≥ N + δ` | `HELMET_DETECTED` |
| R2 | `N ≥ τ` và `N ≥ H + δ` | `SUSPECTED_NO_HELMET` |
| R3 | `H > 0` và `N > 0` nhưng không bên nào trội | `UNKNOWN` (mâu thuẫn) |
| R4 | Chỉ có bằng chứng yếu (< τ) | `UNKNOWN` (bằng chứng yếu) |
| R5 | Không có bằng chứng nào ở vùng đầu | `UNKNOWN` (thiếu bằng chứng) |

`τ` (ngưỡng bằng chứng) và `δ` (biên chênh lệch) được chọn bằng dò lưới trên tập
validation (xem [05_experiments.md](05_experiments.md)).

Trạng thái của **ảnh**: có ít nhất một người nghi vi phạm → `SUSPECTED_NO_HELMET`;
còn người chưa kết luận → `UNKNOWN`; tất cả đội mũ → `HELMET_DETECTED`; không có
người → `NO_PERSON_DETECTED`.

Hệ luật thêm **cờ giải thích** cho các ca khó: `small_person` (khung người cao
< 40 px) và `touches_top_edge` (khung chạm mép trên ảnh, đầu có thể bị cắt).

## 3.5. Lý do lựa chọn phương pháp

* **Không coi "không thấy mũ" là vi phạm.** Mũ có thể bị che khuất, quá nhỏ, mờ
  hoặc nằm ngoài khung hình. Báo vi phạm sai gây mất niềm tin của người dùng
  hơn là báo "chưa rõ". Vì vậy chỉ luật R2 — có bằng chứng *tích cực* là đầu
  trần (`no_helmet`) — mới dẫn tới `SUSPECTED_NO_HELMET`.
* **Luật minh bạch thay vì một mô hình hộp đen thứ hai.** Mỗi kết luận có mã
  luật, lý do và số liệu bằng chứng, dễ giải thích trước giảng viên và dễ kiểm
  tra lỗi.
* **Hình học tương đối thay vì pixel tuyệt đối.** Mọi ngưỡng tính theo kích
  thước khung người nên hoạt động với người ở gần hay xa camera.
* **Ghép một-một toàn cục theo điểm** xử lý được đám đông tốt hơn ghép theo thứ
  tự, nhưng vẫn đơn giản hơn thuật toán Hungarian và cho kết quả giống nhau
  trong hầu hết trường hợp thực tế.
* **Tham số dò trên tập val, báo cáo trên tập test** để tránh "học thuộc" tập
  kiểm tra.

## 3.6. Ảnh đông người và người bị che khuất

Ghép theo khung người (mục 3.3) hoạt động tốt khi người đứng tách nhau, nhưng ảnh
đông người (người đội mũ đứng xen người không đội, che khuất lẫn nhau) có ba lỗi
lặp lại:

1. **Mô hình PPE bỏ sót người bị che** — mũ / đầu trần của họ không có ai để ghép.
2. **Khung người phía trước chứa mũ của người phía sau** — với ghép theo khung,
   người đầu trần đứng trước bị gán nhầm chiếc mũ phía sau → báo "đội mũ" sai.
3. Ở ngưỡng confidence thấp, mô hình bắn `no_helmet` vào **khuôn mặt ngay dưới
   một chiếc mũ** → người đội mũ có thể bị báo nghi vi phạm.

Cách xử lý (không huấn luyện lại YOLO, không đổi dữ liệu):

* **Hợp nhất với mô hình pose** `yolo26n-pose.pt` (huấn luyện sẵn trên COCO, 17
  điểm khớp). Người pose trùng khung người PPE (IoU ≥ 0,5, ghép một-một) được gộp
  làm một và nhận thêm điểm khớp; người chỉ có ở mô hình pose được thêm vào
  (`source = pose`) → giải quyết lỗi 1. Mã: `fusion.fuse_persons`.
* **Ghép theo đầu thật** khi thấy ít nhất hai điểm trên mặt (mũi, mắt, tai):
  tâm đầu = trung bình các điểm mặt; kích thước đầu `s` = lớn nhất của (khoảng
  cách hai tai; 2,4 × hai mắt; 1,2 × mũi–tai). Vì điểm mặt cho `s` hơi nhỏ, dùng
  `s' = max(s, bề rộng vật thể / k)` (k = 1,8 với mũ, 1,3 với đầu trần). Vật thể
  được nhận nếu tâm nằm trong elip quanh vị trí kỳ vọng (mũ cao hơn tâm đầu
  0,6·s', đầu trần 0,3·s'), bán trục 0,9·s' (ngang) và 0,8·s' (dọc), và không
  nhỏ hơn 0,8·s'. Điểm ghép = 1 − 0,5·d². Không thấy mặt (quay lưng) → quay về
  ghép theo khung. Mã: `head.estimate_head`, `association._score_by_head`
  → giải quyết lỗi 2.
* **Bỏ hộp `no_helmet` nằm ngay dưới mũ**: nếu hộp `no_helmet` chồng ≥ 50 % bề
  ngang với một hộp `helmet`, tâm của nó thấp hơn đáy mũ, khoảng hở dọc ≤ nửa
  chiều cao mũ và mũ tin cậy hơn, thì đó là khuôn mặt dưới mũ → loại. Hai hộp
  chồng khít lên nhau thì *không* loại mà để luật R3 (mâu thuẫn) xử lý. Mã:
  `fusion.suppress_faces_under_helmets` → giải quyết lỗi 3.

Mọi tham số mới (bật/tắt pose, ngưỡng người pose, hình elip, bật/tắt bước bỏ
mặt dưới mũ) được dò cùng các ngưỡng cũ trên tập val (mục 5).

**Hạn chế còn lại.** Lớp `no_helmet` của `best.pt` có mAP50 chỉ ~0,11 nên nhiều
người đầu trần trong đám đông vẫn chỉ được kết luận `UNKNOWN` (không có bằng
chứng tích cực) — đây là lựa chọn an toàn theo nguyên tắc "không coi không thấy
mũ là vi phạm". Người *cầm* mũ sát đầu cũng có thể bị đưa về `UNKNOWN`. Muốn
tăng thêm cần huấn luyện lại / bổ sung dữ liệu `no_helmet` trong đám đông.
