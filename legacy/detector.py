
from pathlib import Path
from ultralytics import YOLO


class HelmetDetector:
    """Phat hien cac doi tuong bao ho lao dong bang YOLO."""

    def __init__(self, model_path="models/best.pt", conf=0.25):
        self.model_path = Path(model_path)
        self.conf = conf

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay mo hinh: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(self, image_path):
        """Tra ve danh sach doi tuong phat hien trong anh."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay anh: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": confidence,
                    "bbox": [
                        float(x1), float(y1),
                        float(x2), float(y2)
                    ]
                })

        return detections
