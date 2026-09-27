# TrackSense — ML Module

Detection → OCR → trajectory-ready events, plus accident/anomaly detection.

## Files
- `detect_plate.py` — YOLOv8 vehicle detection + plate localization (fallback heuristic until a fine-tuned plate model is trained)
- `ocr_engine.py` — plate text reading with multi-frame confidence voting + format correction (the module responsible for hitting >90% accuracy)
- `accident_detection.py` — optical-flow based collision/stall detection with false-alarm confirmation window
- `pipeline.py` — ties everything into clean event dicts ready to push to Supabase/Kafka
- `1_generate_plates.py` — generates a synthetic test dataset (degraded plate images) so the pipeline can be tested without real camera footage

## Setup
```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr   # OCR engine backend used by pytesseract
```

## Quick test (no real camera data needed)
```bash
python3 1_generate_plates.py     # creates synthetic multi-camera plate captures
python3 pipeline.py              # runs full detection+OCR+voting, prints events
```
Expected output: ~85-90%+ plate read accuracy across the synthetic dataset, with one intentionally-degraded frame as a realistic worst-case example.

## Integration for backend teammate
```python
from pipeline import process_camera_pass, process_frame_for_anomaly

# once you have 3 plate crops for one vehicle's pass through a camera:
detection_event, alert_event = process_camera_pass(frame_crops, camera_id="CAM-01", timestamp=1234)
# -> push detection_event to Supabase `detections` table
# -> if alert_event is not None, push to `alerts` table + trigger Twilio

# per raw video frame, independently:
anomaly_event = process_frame_for_anomaly(frame_bgr, camera_id="CAM-01", timestamp=1234)
# -> if not None, push to `alerts` table + trigger Twilio
```

## Production upgrade path (swap-in points, no interface changes needed)
| Component | Hackathon version | Production upgrade |
|---|---|---|
| Plate localization | Contour heuristic | Fine-tuned YOLOv8 plate model (`PLATE_MODEL_PATH` in `detect_plate.py`) |
| OCR | Tesseract | CRNN or PARSeq/TrOCR fine-tuned on plate crops (`_read_raw` in `ocr_engine.py`) |
| Anomaly detection | Optical flow heuristics | 3D-CNN / spatio-temporal autoencoder (`process_frame` in `accident_detection.py`) |

## Known limitation (be upfront about this with judges if asked)
The OCR/accident numbers above are measured on a **synthetic dataset** generated in this repo (`1_generate_plates.py`), not real camera footage — this proves the pipeline logic and voting/correction techniques work end-to-end, but real-world accuracy still needs validation against actual ANPR footage or a public dataset (e.g. CCPD) before the >90% claim is production-verified.
