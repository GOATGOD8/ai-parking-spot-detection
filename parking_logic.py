# parking_logic.py

def check_overlap(boxA, boxB, threshold=0.3):
    """
    Check if two boxes overlap enough to consider a parking spot occupied.
    box format: (x, y, width, height)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = boxA[2] * boxA[3]

    if boxA_area == 0:
        return False

    overlap_ratio = inter_area / boxA_area

    return overlap_ratio > threshold


def check_parking(cars, parking_spots):
    """
    Determine which parking spots are occupied.
    
    cars: list of (x, y, w, h)
    parking_spots: list of (x, y, w, h)
    
    returns: list of True/False
    """
    spot_status = []

    for spot in parking_spots:
        occupied = False

        for car in cars:
            if check_overlap(spot, car):
                occupied = True
                break

        spot_status.append(occupied)

    return spot_status


def count_available(spots):
    """
    Count how many spots are free.
    """
    return spots.count(False)


# 🔥 TEST BLOCK (run this file directly to test)
if __name__ == "__main__":
    test_cars = [
        (110, 210, 60, 120),
        (310, 210, 60, 120)
    ]

    test_spots = [
        (100, 200, 80, 160),
        (200, 200, 80, 160),
        (300, 200, 80, 160)
    ]

    status = check_parking(test_cars, test_spots)
    available = count_available(status)

    print("Spot Status:", status)
    print("Available Spots:", available)
