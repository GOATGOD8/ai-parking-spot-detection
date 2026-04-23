import cv2
import numpy as np


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
COLOR_EMPTY    = (0, 220, 100)   # Green  (BGR)
COLOR_OCCUPIED = (0, 60, 220)    # Red    (BGR)
COLOR_CAR_BOX  = (0, 180, 255)   # Orange (BGR)
COLOR_TEXT     = (255, 255, 255) # White  (BGR)
COLOR_OVERLAY  = (0, 0, 0)       # Black  (BGR)

SPOT_THICKNESS = 2
CAR_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL     = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
#  DRAW PARKING SPOTS
# ─────────────────────────────────────────────
def draw_parking_spots(frame, parking_spots):
    """
    Draw each parking spot rectangle on the frame.
    Green = empty, Red = occupied.

    Args:
        frame         : current video frame (numpy array, BGR)
        parking_spots : list of dicts from Person 2, e.g.
                        [{"id": 1, "coords": (x, y, w, h), "occupied": False}, ...]

    Returns:
        frame with parking spots drawn on it
    """
    for spot in parking_spots:
        x, y, w, h = spot["coords"]
        occupied    = spot["occupied"]
        spot_id     = spot["id"]

        color = COLOR_OCCUPIED if occupied else COLOR_EMPTY

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Solid border
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, SPOT_THICKNESS)

        # Spot ID label
        label = f"P{spot_id}"
        (lw, lh), _ = cv2.getTextSize(label, FONT_SMALL, 0.5, 1)
        cv2.rectangle(frame, (x, y), (x + lw + 6, y + lh + 6), color, -1)
        cv2.putText(frame, label, (x + 3, y + lh + 2),
                    FONT_SMALL, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

        # Status dot
        dot_x = x + w - 10
        dot_y = y + 10
        cv2.circle(frame, (dot_x, dot_y), 5, color, -1)
        cv2.circle(frame, (dot_x, dot_y), 5, COLOR_TEXT, 1)

    return frame


# ─────────────────────────────────────────────
#  DRAW CAR BOUNDING BOXES  (from YOLO)
# ─────────────────────────────────────────────
def draw_car_boxes(frame, car_boxes):
    """
    Draw YOLO detection bounding boxes for each detected car.

    Args:
        frame     : current video frame (numpy array, BGR)
        car_boxes : list of tuples from Person 1, e.g. [(x, y, w, h), ...]

    Returns:
        frame with car boxes drawn on it
    """
    for idx, (x, y, w, h) in enumerate(car_boxes):
        # Dashed-style box using corner brackets
        bracket = 14  # length of corner lines

        corners = [
            # top-left
            [(x, y + bracket), (x, y), (x + bracket, y)],
            # top-right
            [(x + w - bracket, y), (x + w, y), (x + w, y + bracket)],
            # bottom-left
            [(x, y + h - bracket), (x, y + h), (x + bracket, y + h)],
            # bottom-right
            [(x + w - bracket, y + h), (x + w, y + h), (x + w, y + h - bracket)],
        ]

        for pts in corners:
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], COLOR_CAR_BOX, CAR_THICKNESS, cv2.LINE_AA)

        # "CAR" label above the box
        label = f"CAR {idx + 1}"
        (lw, lh), _ = cv2.getTextSize(label, FONT_SMALL, 0.45, 1)
        label_y = max(y - 6, lh + 4)
        cv2.rectangle(frame, (x, label_y - lh - 4), (x + lw + 6, label_y + 2), COLOR_CAR_BOX, -1)
        cv2.putText(frame, label, (x + 3, label_y - 2),
                    FONT_SMALL, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
#  DRAW HUD COUNTER
# ─────────────────────────────────────────────
def draw_counter(frame, parking_spots):
    """
    Draw a HUD-style availability counter in the top-left corner.

    Args:
        frame         : current video frame (numpy array, BGR)
        parking_spots : list of dicts from Person 2

    Returns:
        frame with counter drawn on it
    """
    total     = len(parking_spots)
    available = sum(1 for s in parking_spots if not s["occupied"])
    occupied  = total - available

    color = COLOR_EMPTY if available > 0 else COLOR_OCCUPIED

    # Background panel
    panel_x, panel_y = 12, 12
    panel_w, panel_h = 230, 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent bar on left side
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + 4, panel_y + panel_h), color, -1)

    # Main counter text
    main_text = f"AVAILABLE: {available}/{total}"
    cv2.putText(frame, main_text, (panel_x + 14, panel_y + 32),
                FONT, 0.7, color, 1, cv2.LINE_AA)

    # Sub-line
    sub_text = f"Occupied: {occupied}   Free: {available}"
    cv2.putText(frame, sub_text, (panel_x + 14, panel_y + 60),
                FONT_SMALL, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

    return frame


# ─────────────────────────────────────────────
#  MASTER RENDER FUNCTION  ← call this in main.py
# ─────────────────────────────────────────────
def render_frame(frame, parking_spots, car_boxes):
    """
    Master function — call once per video frame.
    Combines all visual layers: parking spots, car boxes, and HUD counter.

    Args:
        frame         : raw video frame (numpy array, BGR)
        parking_spots : list of dicts from Person 2
                        [{"id": int, "coords": (x,y,w,h), "occupied": bool}, ...]
        car_boxes     : list of tuples from Person 1
                        [(x, y, w, h), ...]

    Returns:
        annotated frame ready to display with cv2.imshow()
    """
    frame = draw_parking_spots(frame, parking_spots)
    frame = draw_car_boxes(frame, car_boxes)
    frame = draw_counter(frame, parking_spots)
    return frame


# ─────────────────────────────────────────────
#  STANDALONE TEST  (run: python ui.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a parking lot frame
    frame = np.full((480, 720, 3), (50, 50, 50), dtype=np.uint8)

    # Draw faint lane lines for realism
    for i in range(0, 720, 80):
        cv2.line(frame, (i, 0), (i, 480), (65, 65, 65), 1)
    for i in range(0, 480, 80):
        cv2.line(frame, (0, i), (720, i), (65, 65, 65), 1)

    # Fake parking spots (replace with Person 2's output in main.py)
    fake_spots = [
        {"id": 1, "coords": (40,  150, 110, 65), "occupied": False},
        {"id": 2, "coords": (170, 150, 110, 65), "occupied": True},
        {"id": 3, "coords": (300, 150, 110, 65), "occupied": False},
        {"id": 4, "coords": (430, 150, 110, 65), "occupied": True},
        {"id": 5, "coords": (560, 150, 110, 65), "occupied": False},
        {"id": 6, "coords": (40,  280, 110, 65), "occupied": False},
        {"id": 7, "coords": (170, 280, 110, 65), "occupied": True},
        {"id": 8, "coords": (300, 280, 110, 65), "occupied": False},
    ]

    # Fake YOLO car boxes (replace with Person 1's output in main.py)
    fake_cars = [
        (165, 145, 120, 75),
        (425, 145, 120, 75),
        (165, 275, 120, 75),
    ]

    result = render_frame(frame, fake_spots, fake_cars)

    cv2.imshow("Person 3 — UI Test (press Q to quit)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("UI test complete. ui.py is ready for integration into main.py.")
