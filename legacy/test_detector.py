
from detector import HelmetDetector

detector = HelmetDetector()

image_path = "data/raw/anh_ct_nh (1).jpg"

detections = detector.detect(image_path)

print(f"So doi tuong phat hien: {len(detections)}")

for item in detections:
    print(
        f"Lop: {item['class_name']}, "
        f"confidence: {item['confidence']:.2f}, "
        f"bbox: {item['bbox']}"
    )
