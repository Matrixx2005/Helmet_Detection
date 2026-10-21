
from ultralytics import YOLO


def main():
    # Tai mo hinh YOLO nho
    model = YOLO("yolo11n.pt")

    # Phat hien doi tuong trong anh mau
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        imgsz=640,
        conf=0.25,
        save=True,
        device="cpu"
    )

    # In ket qua nhan dien
    for result in results:
        print("\nCac doi tuong phat hien:")

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            print(
                f"{class_name}: "
                f"confidence = {confidence:.2f}"
            )

    print("\nDa chay xong YOLO!")


if __name__ == "__main__":
    main()
