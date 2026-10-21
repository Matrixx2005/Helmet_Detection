
from pathlib import Path
from ultralytics import YOLO

# Duong dan mo hinh va anh thu nghiem
model_path = Path("models/best.pt")
image_path = Path("data/raw/anh_ct_nh (1).jpg")

# Kiem tra file
if not model_path.exists():
    raise FileNotFoundError(f"Khong tim thay mo hinh: {model_path}")

if not image_path.exists():
    raise FileNotFoundError(f"Khong tim thay anh: {image_path}")

# Nap mo hinh da huan luyen
model = YOLO(str(model_path))

print("Cac lop mo hinh nhan dien:")
print(model.names)

# Nhan dien anh
results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True,
    project="outputs",
    name="helmet_test",
    exist_ok=True
)

# In ket qua
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"{class_name}: confidence = {confidence:.2f}")

print("Da nhan dien xong!")
