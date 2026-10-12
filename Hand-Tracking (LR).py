
import cv2
import mediapipe as mp
import math
import time
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 21 hand landmarks and their connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


class HandDetector:
    def __init__(
        self, mode=False, maxHands=2,
        detectionCon=0.5, minTrackCon=0.5
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.tipIds = [4, 8, 12, 16, 20]

        model_path = Path(__file__).resolve().parent / "hand_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {model_path}\n"
                "Hãy tải hand_landmarker.task vào cùng thư mục với file Python."
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=detectionCon,
            min_tracking_confidence=minTrackCon
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.results = None

    def findHands(self, img, draw=True, flipType=True):
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        # VIDEO mode requires increasing timestamps
        self.timestamp_ms += 33
        self.results = self.detector.detect_for_video(
            mp_image, self.timestamp_ms
        )

        allHands = []

        if self.results.hand_landmarks:
            for i, landmarks in enumerate(self.results.hand_landmarks):
                points = []
                xList, yList = [], []

                for lm in landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append([px, py])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)

                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                center = (
                    xmin + (xmax - xmin) // 2,
                    ymin + (ymax - ymin) // 2
                )

                label = self.results.handedness[i][0].category_name
                if flipType:
                    label = "Left" if label == "Right" else "Right"

                hand = {
                    "lmList": points,
                    "bbox": bbox,
                    "center": center,
                    "type": label
                }
                allHands.append(hand)

                if draw:
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(img, tuple(points[a]), tuple(points[b]),
                                 (0, 255, 0), 2)

                    for px, py in points:
                        cv2.circle(img, (px, py), 4, (255, 0, 255), -1)

                    cv2.rectangle(
                        img,
                        (xmin - 20, ymin - 20),
                        (xmax + 20, ymax + 20),
                        (255, 0, 255), 2
                    )
                    cv2.putText(
                        img, label, (xmin - 10, ymin - 25),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 2
                    )

        return (allHands, img) if draw else allHands

    def fingersUp(self, myHand):
        points = myHand["lmList"]
        handType = myHand["type"]
        fingers = []

        # Thumb
        if handType == "Right":
            fingers.append(int(points[4][0] > points[3][0]))
        else:
            fingers.append(int(points[4][0] < points[3][0]))

        # Index, middle, ring and little fingers
        for tip in self.tipIds[1:]:
            fingers.append(int(points[tip][1] < points[tip - 2][1]))

        return fingers

    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), -1)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), -1)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), -1)
            return length, info, img

        return length, info

    def close(self):
        self.detector.close()


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không mở được webcam. Hãy kiểm tra camera và quyền truy cập.")
        return

    detector = HandDetector(detectionCon=0.8, maxHands=2)

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Không đọc được hình ảnh từ webcam.")
                break

            hands, img = detector.findHands(img)

            if len(hands) >= 1:
                hand1 = hands[0]
                fingers1 = detector.fingersUp(hand1)

                cv2.putText(
                    img, f"Fingers: {sum(fingers1)}",
                    (10, 40), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0), 2
                )

            if len(hands) >= 2:
                hand2 = hands[1]
                length, info, img = detector.findDistance(
                    hands[0]["lmList"][8],
                    hand2["lmList"][8],
                    img
                )

                cv2.putText(
                    img, f"Distance: {int(length)} px",
                    (10, 75), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2
                )

            cv2.imshow("Hand Tracking - Left/Right", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
