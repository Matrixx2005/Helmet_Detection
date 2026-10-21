
from detector import HelmetDetector
from association import associate_person_helmet

detector = HelmetDetector()
image_path = "data/raw/anh_ct_nh (1).jpg"

detections = detector.detect(image_path)
associations = associate_person_helmet(detections)

print("\n=== KIEM TRA LIEN KET NGUOI - MU ===")

for item in associations:
    print(f"\nPerson #{item['person_index']}")
    print(f"  Person bbox: {item['person_bbox']}")
    print(f"  Co mu lien ket: {item['helmet_detected']}")
    print(f"  Helmet bbox: {item['helmet_bbox']}")

    if item.get("no_helmet_evidence", False):
        print("  Co bang chung no_helmet: True")
    else:
        print("  Co bang chung no_helmet: False")
