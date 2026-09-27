"""
Stage 1 of the ANPR pipeline: vehicle + license-plate detection.

- Uses YOLOv8 (pretrained on COCO) to detect vehicles in a full camera frame.
- Each vehicle crop is passed on for plate localization.
- NOTE: For production, replace `PLATE_MODEL_PATH` with a YOLOv8 model
  fine-tuned specifically on license-plate bounding boxes (e.g. trained on a
  Roboflow Universe plate dataset, ~200-500 labeled images is enough for a
  hackathon-grade fine-tune). Until that's trained, this file falls back to
  a simple contour-based plate localizer within each vehicle crop, so the
  rest of the pipeline can be built and tested end-to-end in parallel.
"""
from ultralytics import YOLO
import cv2
import numpy as np

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PLATE_MODEL_PATH = None  # set to "plate_yolov8.pt" once your fine-tuned model is ready

_vehicle_model = None
_plate_model = None


def _load_vehicle_model():
    global _vehicle_model
    if _vehicle_model is None:
        _vehicle_model = YOLO("yolov8n.pt")  # auto-downloads pretrained weights on first run
    return _vehicle_model


def _load_plate_model():
    global _plate_model
    if PLATE_MODEL_PATH and _plate_model is None:
        _plate_model = YOLO(PLATE_MODEL_PATH)
    return _plate_model


def detect_vehicles(frame_bgr, conf_thresh=0.35):
    """Returns list of {bbox:(x1,y1,x2,y2), class, confidence} for each vehicle in the frame."""
    model = _load_vehicle_model()
    results = model.predict(frame_bgr, conf=conf_thresh, verbose=False)[0]
    vehicles = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            vehicles.append({
                "bbox": (x1, y1, x2, y2),
                "class": VEHICLE_CLASSES[cls_id],
                "confidence": float(box.conf[0]),
            })
    return vehicles


def _fallback_plate_locate(vehicle_crop_bgr):
    """
    Heuristic plate localizer (used until a fine-tuned plate model is available):
    looks for a high-contrast, roughly rectangular region in the lower half of
    the vehicle crop, which is where plates usually sit in typical ANPR angles.
    """
    h, w = vehicle_crop_bgr.shape[:2]
    lower_half = vehicle_crop_bgr[int(h * 0.5):, :]
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 40 or ch < 12:
            continue
        aspect = cw / max(ch, 1)
        if 2.0 <= aspect <= 6.0:  # plates are wide rectangles
            score = cw * ch
            if score > best_score:
                best_score = score
                best = (x, y + int(h * 0.5), x + cw, y + int(h * 0.5) + ch)
    return best  # bbox within vehicle_crop coords, or None if nothing found


def detect_plate_region(vehicle_crop_bgr):
    """Returns a cropped plate image (bgr np.array) or None if no plate was localized."""
    model = _load_plate_model()
    if model is not None:
        results = model.predict(vehicle_crop_bgr, conf=0.4, verbose=False)[0]
        if len(results.boxes) > 0:
            box = results.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            return vehicle_crop_bgr[y1:y2, x1:x2]
        return None

    bbox = _fallback_plate_locate(vehicle_crop_bgr)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return vehicle_crop_bgr[y1:y2, x1:x2]


def rectify_plate(plate_crop_bgr):
    """Perspective-correct an angled plate crop to a frontal view before OCR."""
    gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop_bgr

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.array(sorted(box, key=lambda p: (p[1], p[0])), dtype="float32")

    w, h = plate_crop_bgr.shape[1], plate_crop_bgr.shape[0]
    dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype="float32")
    try:
        M = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(plate_crop_bgr, M, (w, h))
        return warped
    except cv2.error:
        return plate_crop_bgr


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is not None:
        vehicles = detect_vehicles(img)
        print(f"Detected {len(vehicles)} vehicle(s): {vehicles}")
    else:
        print("Usage: python detect_plate.py <path_to_traffic_scene.jpg>")
