from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8n.pt")

def detect_cars(frame):
    """
    Detect vehicles in a frame and return bounding boxes
    in the format (x, y, w, h).
    """
    results = model(frame)
    car_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO classes:
            # 2 = car, 5 = bus, 7 = truck
            if cls in [2, 5, 7] and conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0]

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                car_boxes.append((x, y, w, h))

    return car_boxes
