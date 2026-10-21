
from detector import HelmetDetector
from association import associate_person_helmet
from rule_engine import infer_safety_status

detector = HelmetDetector()
image_path = "data/raw/anh_ct_nh (1).jpg"

detections = detector.detect(image_path)
associations = associate_person_helmet(detections)
results = infer_safety_status(detections, associations)

print("\n=== KET QUA SUY LUAN ===")

for result in results:
    print(f"Nguoi: {result['person_index']}")
    print(f"Trang thai: {result['status']}")
    print(f"Giai thich: {result['reason']}")
    print("-" * 35)
